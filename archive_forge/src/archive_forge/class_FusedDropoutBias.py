from typing import Optional
import torch
import triton
from torch.cuda.amp import custom_bwd, custom_fwd
from xformers.components.activations import Activation, build_activation
from xformers.triton.k_activations import get_triton_activation_index
from xformers.triton.k_dropout import k_dropout_bw, k_dropout_fw
class FusedDropoutBias(torch.nn.Module):
    """
    A layer which fuses the computation of Dropout(Activation(x))
    in a single GPU kernel
    """

    def __init__(self, p: float, bias_shape: Optional[int], activation: Optional[Activation]=None) -> None:
        super().__init__()
        self.p = float(p)
        assert self.p < 1.0, f"We don't want to drop all the values, most probably p={p} is not properly set"
        self.activation_type = activation
        self.bias = torch.zeros(bias_shape, requires_grad=True) if bias_shape is not None else None
        self.activation = get_triton_activation_index(self.activation_type)
        self.activation_pytorch = build_activation(self.activation_type)

    def init_weights(self, *args, **kwargs):
        with torch.no_grad():
            if self.bias is not None:
                self.bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            self.bias = self.bias.to(dtype=x.dtype, device=x.device)
        p = self.p if self.training else 0.0
        perf_check = x.shape[-1] > 512
        if not x.is_cuda or not perf_check or p == 0.0:
            x = x + self.bias if self.bias is not None else x
            x = self.activation_pytorch(x)
            return torch.nn.functional.dropout(x, p) if p > 0.0 else x
        return _dropout.apply(x, p, self.bias, self.activation, True)