from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def _fft_transform(dense: Tensor, dim: int) -> Tensor:
    """Wrapper of torch.fft.fft with more flexibility on dimensions.

    TODO (Min): figure out if we need to change other args like frequency length, n, or
                the normalization flag.

    For our use case, we use fft not rfft since we want big magnitute components from
    both positive and negative frequencies.

    Args:
        dense (Tensor):
            Input dense tensor (no zeros).
        dim (int):
            Which dimension to transform.
    Returns:
        (Tensor, complex):
            transformed dense tensor FFT components.
    """
    orig_shape = None
    if dim is None:
        orig_shape = dense.shape
        dense = dense.reshape(-1)
        dim = -1
    ret = torch.fft.fft(dense, dim=dim)
    if orig_shape is not None:
        ret = ret.reshape(orig_shape)
    return ret