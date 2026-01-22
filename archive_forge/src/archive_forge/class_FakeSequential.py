from typing import Tuple
import torch
import torch.nn as nn
class FakeSequential(nn.Module):

    def __init__(self, *modules: Tuple[nn.Module, ...]) -> None:
        super().__init__()
        self._module_sequence = list(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self._module_sequence:
            x = module(x)
        return x