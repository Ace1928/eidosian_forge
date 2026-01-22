import math
from dataclasses import dataclass
from enum import Enum
import torch
class PatchToSequence(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.flatten(2, 3).transpose(1, 2).contiguous()