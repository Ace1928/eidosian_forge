from functools import partial
from typing import Optional
import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup
from flash_attn.ops.activations import gelu_bwd, relu_bwd, sqrelu_bwd, sqrelu_fwd
from flash_attn.utils.distributed import (
class FusedDense(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool=True, return_residual: bool=False, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.return_residual = return_residual

    def forward(self, x, process_group=None):
        """
        If process_group is not None, we're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul.
        """
        return fused_dense_func(x, self.weight, self.bias, return_residual=self.return_residual, process_group=process_group)