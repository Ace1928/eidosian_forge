import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class Deemphasis(torch.nn.Module):
    """De-emphasizes a waveform along its last dimension.
    See :meth:`torchaudio.functional.deemphasis` for more details.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        coeff (float, optional): De-emphasis coefficient. Typically between 0.0 and 1.0.
            (Default: 0.97)
    """

    def __init__(self, coeff: float=0.97) -> None:
        super().__init__()
        self.coeff = coeff

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (torch.Tensor): Waveform, with shape `(..., N)`.

        Returns:
            torch.Tensor: De-emphasized waveform, with shape `(..., N)`.
        """
        return F.deemphasis(waveform, coeff=self.coeff)