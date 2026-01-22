import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class MuLawDecoding(torch.nn.Module):
    """Decode mu-law encoded signal.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and ``quantization_channels - 1``
    and returns a signal scaled between -1 and 1.

    Args:
        quantization_channels (int, optional): Number of channels. (Default: ``256``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = torchaudio.transforms.MuLawDecoding(quantization_channels=512)
        >>> mulawtrans = transform(waveform)
    """
    __constants__ = ['quantization_channels']

    def __init__(self, quantization_channels: int=256) -> None:
        super(MuLawDecoding, self).__init__()
        self.quantization_channels = quantization_channels

    def forward(self, x_mu: Tensor) -> Tensor:
        """
        Args:
            x_mu (Tensor): A mu-law encoded signal which needs to be decoded.

        Returns:
            Tensor: The signal decoded.
        """
        return F.mu_law_decoding(x_mu, self.quantization_channels)