import math
from dataclasses import dataclass
from typing import (
import torch
class LowerTriangularMaskWithTensorBias(LowerTriangularMask):
    """A lower-triangular (aka causal) mask with an additive bias"""

    def __init__(self, bias: torch.Tensor) -> None:
        self._bias = bias

    def materialize(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        return super().materialize(shape, dtype=dtype, device=device) + self._bias