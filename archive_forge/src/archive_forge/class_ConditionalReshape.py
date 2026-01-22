import math
from dataclasses import dataclass
from enum import Enum
import torch
class ConditionalReshape(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        if x.ndim == 3:
            B, HW, C = x.shape
            H = int(math.sqrt(HW))
            assert H * H == HW, f'{(H, HW)}'
            x = x.transpose(1, 2).reshape(B, C, H, H)
        return x