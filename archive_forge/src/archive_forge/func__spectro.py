import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
def _spectro(x: torch.Tensor, n_fft: int=512, hop_length: int=0, pad: int=0) -> torch.Tensor:
    other = list(x.shape[:-1])
    length = int(x.shape[-1])
    x = x.reshape(-1, length)
    z = torch.stft(x, n_fft * (1 + pad), hop_length, window=torch.hann_window(n_fft).to(x), win_length=n_fft, normalized=True, center=True, return_complex=True, pad_mode='reflect')
    _, freqs, frame = z.shape
    other.extend([freqs, frame])
    return z.view(other)