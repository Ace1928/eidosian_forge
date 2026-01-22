import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class ComputeDeltas(torch.nn.Module):
    """Compute delta coefficients of a tensor, usually a spectrogram.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    See `torchaudio.functional.compute_deltas` for more details.

    Args:
        win_length (int, optional): The window length used for computing delta. (Default: ``5``)
        mode (str, optional): Mode parameter passed to padding. (Default: ``"replicate"``)
    """
    __constants__ = ['win_length']

    def __init__(self, win_length: int=5, mode: str='replicate') -> None:
        super(ComputeDeltas, self).__init__()
        self.win_length = win_length
        self.mode = mode

    def forward(self, specgram: Tensor) -> Tensor:
        """
        Args:
            specgram (Tensor): Tensor of audio of dimension (..., freq, time).

        Returns:
            Tensor: Tensor of deltas of dimension (..., freq, time).
        """
        return F.compute_deltas(specgram, win_length=self.win_length, mode=self.mode)