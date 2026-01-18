from typing import Optional, Sequence
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import divide
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.utils import set_weight_attrs
def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
    parallel_dim = param.parallel_dim
    assert loaded_weight.shape[parallel_dim] == self.org_vocab_size
    loaded_weight = loaded_weight[self.vocab_start_index:self.vocab_end_index]
    param[:loaded_weight.shape[0]].data.copy_(loaded_weight)