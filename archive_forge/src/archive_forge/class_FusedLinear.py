import math
from typing import Any, Optional
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd
from xformers.components.activations import Activation
from xformers.triton.k_activations import get_triton_activation_index
from xformers.triton.k_fused_matmul_bw import fused_matmul_backward
from xformers.triton.k_fused_matmul_fw import fused_matmul
class FusedLinear(nn.Module):
    """
    Handle a linear transform, like torch.nn.Linear_, and a given activation, in a single kernel.
    The whole transform: is :math:`y = activation(xA^T + b)`.

    This is typically significantly faster than PyTorch while using fp16 and non-sigmoid activations,
    as of September 2021.

    .. _torch.nn.Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """

    def __init__(self, in_features: int, out_features: int, bias: bool=False, activation: Optional[Activation]=None, **_):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=True) if bias else None
        self._activation_index = get_triton_activation_index(activation)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _fused_linear_triton.apply(x, self.weight, self.bias, self._activation_index, self.weight.requires_grad, self.bias.requires_grad if self.bias is not None else False, self.training and x.requires_grad and (self._activation_index > 0))