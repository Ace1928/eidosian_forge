from typing import List, Tuple
import torch
from torch import Tensor
from .phony import get_phony
class Fork(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Fork', input: Tensor) -> Tuple[Tensor, Tensor]:
        phony = get_phony(input.device, requires_grad=False)
        return (input.detach(), phony.detach())

    @staticmethod
    def backward(ctx: 'Fork', grad_input: Tensor, grad_grad: Tensor) -> Tensor:
        return grad_input