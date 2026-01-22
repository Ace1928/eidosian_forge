import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class SpeedPerturbation(torch.nn.Module):
    """Applies the speed perturbation augmentation introduced in
    *Audio augmentation for speech recognition* :cite:`ko15_interspeech`. For a given input,
    the module samples a speed-up factor from ``factors`` uniformly at random and adjusts
    the speed of the input by that factor.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        orig_freq (int): Original frequency of the signals in ``waveform``.
        factors (Sequence[float]): Factors by which to adjust speed of input. Values greater than 1.0
            compress ``waveform`` in time, whereas values less than 1.0 stretch ``waveform`` in time.

    Example
        >>> speed_perturb = SpeedPerturbation(16000, [0.9, 1.1, 1.0, 1.0, 1.0])
        >>> # waveform speed will be adjusted by factor 0.9 with 20% probability,
        >>> # 1.1 with 20% probability, and 1.0 (i.e. kept the same) with 60% probability.
        >>> speed_perturbed_waveform = speed_perturb(waveform, lengths)
    """

    def __init__(self, orig_freq: int, factors: Sequence[float]) -> None:
        super().__init__()
        self.speeders = torch.nn.ModuleList([Speed(orig_freq=orig_freq, factor=factor) for factor in factors])

    def forward(self, waveform: torch.Tensor, lengths: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            waveform (torch.Tensor): Input signals, with shape `(..., time)`.
            lengths (torch.Tensor or None, optional): Valid lengths of signals in ``waveform``, with shape `(...)`.
                If ``None``, all elements in ``waveform`` are treated as valid. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor or None):
                torch.Tensor
                    Speed-adjusted waveform, with shape `(..., new_time).`
                torch.Tensor or None
                    If ``lengths`` is not ``None``, valid lengths of signals in speed-adjusted waveform,
                    with shape `(...)`; otherwise, ``None``.
        """
        idx = int(torch.randint(len(self.speeders), ()))
        for speeder_idx, speeder in enumerate(self.speeders):
            if idx == speeder_idx:
                return speeder(waveform, lengths)
        raise RuntimeError('Speeder not found; execution should have never reached here.')