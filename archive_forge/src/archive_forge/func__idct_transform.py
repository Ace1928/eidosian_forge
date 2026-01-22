from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def _idct_transform(sst: Tensor, dim: int) -> Tensor:
    """Should take a tensor and perform an inverse Discrete Cosine Transform and return a new tensor.

    Args:
        sst (Tensor):
            Input sst tensor (may have zeros) in frequency domain.
        dim (int):
            Which dimension to transform.
    Returns:
        (Tensor):
            A new, transformed dense tensor with real domain values.
    """
    raise NotImplementedError('Support for iDCT has not been implemented yet!')