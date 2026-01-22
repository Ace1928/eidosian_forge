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
def _apply_convolve_mode(conv_result: torch.Tensor, x_length: int, y_length: int, mode: str) -> torch.Tensor:
    valid_convolve_modes = ['full', 'valid', 'same']
    if mode == 'full':
        return conv_result
    elif mode == 'valid':
        target_length = max(x_length, y_length) - min(x_length, y_length) + 1
        start_idx = (conv_result.size(-1) - target_length) // 2
        return conv_result[..., start_idx:start_idx + target_length]
    elif mode == 'same':
        start_idx = (conv_result.size(-1) - x_length) // 2
        return conv_result[..., start_idx:start_idx + x_length]
    else:
        raise ValueError(f"Unrecognized mode value '{mode}'. Please specify one of {valid_convolve_modes}.")