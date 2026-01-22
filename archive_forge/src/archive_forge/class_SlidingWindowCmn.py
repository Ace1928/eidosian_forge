import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class SlidingWindowCmn(torch.nn.Module):
    """
    Apply sliding-window cepstral mean (and optionally variance) normalization per utterance.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        cmn_window (int, optional): Window in frames for running average CMN computation (int, default = 600)
        min_cmn_window (int, optional):  Minimum CMN window used at start of decoding (adds latency only at start).
            Only applicable if center == false, ignored if center==true (int, default = 100)
        center (bool, optional): If true, use a window centered on the current frame
            (to the extent possible, modulo end effects). If false, window is to the left. (bool, default = false)
        norm_vars (bool, optional): If true, normalize variance to one. (bool, default = false)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.SlidingWindowCmn(cmn_window=1000)
        >>> cmn_waveform = transform(waveform)
    """

    def __init__(self, cmn_window: int=600, min_cmn_window: int=100, center: bool=False, norm_vars: bool=False) -> None:
        super().__init__()
        self.cmn_window = cmn_window
        self.min_cmn_window = min_cmn_window
        self.center = center
        self.norm_vars = norm_vars

    def forward(self, specgram: Tensor) -> Tensor:
        """
        Args:
            specgram (Tensor): Tensor of spectrogram of dimension `(..., time, freq)`.

        Returns:
            Tensor: Tensor of spectrogram of dimension `(..., time, freq)`.
        """
        cmn_specgram = F.sliding_window_cmn(specgram, self.cmn_window, self.min_cmn_window, self.center, self.norm_vars)
        return cmn_specgram