from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (
from vllm.sequence import SamplerOutput
from hf_olmo import OLMoConfig
class OlmoModel(nn.Module):

    def __init__(self, config: OLMoConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(wte=VocabParallelEmbedding(config.embedding_size or config.vocab_size, config.d_model), ln_f=nn.LayerNorm(config.d_model, elementwise_affine=False, bias=False)))
        blocks = [OlmoBlock(config, linear_method) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            raise NotImplementedError('Block group size > 1 not supported yet')
        else:
            self.transformer.update({'blocks': nn.ModuleList(blocks)})
        if not config.weight_tying:
            self.transformer.update({'ff_out': ColumnParallelLinear(config.d_model, config.embedding_size or config.vocab_size, bias=config.include_bias, linear_method=linear_method)})

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata) -> torch.Tensor:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        x = self.transformer.wte(input_ids)
        for block_idx, block in enumerate(self.transformer.blocks):
            x = block(positions, x, kv_caches[block_idx], input_metadata)
        x = self.transformer.ln_f(x)
        return x