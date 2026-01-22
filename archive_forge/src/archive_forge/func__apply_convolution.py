from typing import Optional, Tuple
import torch
def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
    residual = input
    input = input.transpose(0, 1)
    input = self.conv_module(input)
    input = input.transpose(0, 1)
    input = residual + input
    return input