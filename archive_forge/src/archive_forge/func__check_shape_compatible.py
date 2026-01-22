import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated
from .filtering import highpass_biquad, treble_biquad
def _check_shape_compatible(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.ndim != y.ndim:
        raise ValueError(f'The operands must be the same dimension (got {x.ndim} and {y.ndim}).')
    for i in range(x.ndim - 1):
        xi = x.size(i)
        yi = y.size(i)
        if xi == yi or xi == 1 or yi == 1:
            continue
        raise ValueError(f'Leading dimensions of x and y are not broadcastable (got {x.shape} and {y.shape}).')