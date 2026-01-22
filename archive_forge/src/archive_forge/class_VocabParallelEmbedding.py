from typing import Optional, Sequence
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import divide
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.utils import set_weight_attrs
class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, params_dtype: Optional[torch.dtype]=None, org_num_embeddings: Optional[int]=None, padding_size: int=DEFAULT_VOCAB_PADDING_SIZE):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.org_vocab_size = org_num_embeddings or num_embeddings
        self.num_embeddings_padded = pad_vocab_size(num_embeddings, padding_size)
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.vocab_start_index, self.vocab_end_index = vocab_range_from_global_vocab_size(self.num_embeddings_padded, get_tensor_model_parallel_rank(), self.tp_size)
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.weight = Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim, dtype=params_dtype))
        set_weight_attrs(self.weight, {'parallel_dim': 0, 'weight_loader': self.weight_loader})

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        parallel_dim = param.parallel_dim
        assert loaded_weight.shape[parallel_dim] == self.org_vocab_size
        loaded_weight = loaded_weight[self.vocab_start_index:self.vocab_end_index]
        param[:loaded_weight.shape[0]].data.copy_(loaded_weight)

    def forward(self, input_):
        if self.tp_size > 1:
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        output_parallel = F.embedding(masked_input, self.weight)
        if self.tp_size > 1:
            output_parallel[input_mask, :] = 0.0
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output