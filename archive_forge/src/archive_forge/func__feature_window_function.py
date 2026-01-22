import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
def _feature_window_function(window_type: str, window_size: int, blackman_coeff: float, device: torch.device, dtype: int) -> Tensor:
    """Returns a window function with the given type and size"""
    if window_type == HANNING:
        return torch.hann_window(window_size, periodic=False, device=device, dtype=dtype)
    elif window_type == HAMMING:
        return torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46, device=device, dtype=dtype)
    elif window_type == POVEY:
        return torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85)
    elif window_type == RECTANGULAR:
        return torch.ones(window_size, device=device, dtype=dtype)
    elif window_type == BLACKMAN:
        a = 2 * math.pi / (window_size - 1)
        window_function = torch.arange(window_size, device=device, dtype=dtype)
        return (blackman_coeff - 0.5 * torch.cos(a * window_function) + (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)).to(device=device, dtype=dtype)
    else:
        raise Exception('Invalid window type ' + window_type)