import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class MuLawEncoding(torch.nn.Module):
    """Encode signal based on mu-law companding.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1

    Args:
        quantization_channels (int, optional): Number of channels. (Default: ``256``)

    Example
       >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
       >>> transform = torchaudio.transforms.MuLawEncoding(quantization_channels=512)
       >>> mulawtrans = transform(waveform)

    """
    __constants__ = ['quantization_channels']

    def __init__(self, quantization_channels: int=256) -> None:
        super(MuLawEncoding, self).__init__()
        self.quantization_channels = quantization_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): A signal to be encoded.

        Returns:
            Tensor: An encoded signal.
        """
        return F.mu_law_encoding(x, self.quantization_channels)