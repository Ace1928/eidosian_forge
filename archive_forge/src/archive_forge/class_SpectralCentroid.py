import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class SpectralCentroid(torch.nn.Module):
    """Compute the spectral centroid for each channel along the time axis.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    The spectral centroid is defined as the weighted average of the
    frequency values, weighted by their magnitude.

    Args:
        sample_rate (int): Sample rate of audio signal.
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.SpectralCentroid(sample_rate)
        >>> spectral_centroid = transform(waveform)  # (channel, time)
    """
    __constants__ = ['sample_rate', 'n_fft', 'win_length', 'hop_length', 'pad']

    def __init__(self, sample_rate: int, n_fft: int=400, win_length: Optional[int]=None, hop_length: Optional[int]=None, pad: int=0, window_fn: Callable[..., Tensor]=torch.hann_window, wkwargs: Optional[dict]=None) -> None:
        super(SpectralCentroid, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of dimension `(..., time)`.

        Returns:
            Tensor: Spectral Centroid of size `(..., time)`.
        """
        return F.spectral_centroid(waveform, self.sample_rate, self.pad, self.window, self.n_fft, self.hop_length, self.win_length)