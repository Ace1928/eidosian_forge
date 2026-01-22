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
def _median_smoothing(indices: Tensor, win_length: int) -> Tensor:
    """
    Apply median smoothing to the 1D tensor over the given window.
    """
    pad_length = (win_length - 1) // 2
    indices = torch.nn.functional.pad(indices, (pad_length, 0), mode='constant', value=0.0)
    indices[..., :pad_length] = torch.cat(pad_length * [indices[..., pad_length].unsqueeze(-1)], dim=-1)
    roll = indices.unfold(-1, win_length, 1)
    values, _ = torch.median(roll, -1)
    return values