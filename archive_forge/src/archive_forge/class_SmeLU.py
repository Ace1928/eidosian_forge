from enum import Enum
from typing import Optional
import torch
from torch import nn
class SmeLU(nn.Module):

    def __init__(self, beta: float=2.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu = torch.where(x >= self.beta, x, torch.tensor([0.0], device=x.device, dtype=x.dtype))
        return torch.where(torch.abs(x) <= self.beta, ((x + self.beta) ** 2).type_as(x) / (4.0 * self.beta), relu)