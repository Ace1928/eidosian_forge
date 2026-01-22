import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class TimeStretch(torch.nn.Module):
    """Stretch stft in time without modifying pitch for a given rate.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Proposed in *SpecAugment* :cite:`specaugment`.

    Args:
        hop_length (int or None, optional): Length of hop between STFT windows.
            (Default: ``n_fft // 2``, where ``n_fft == (n_freq - 1) * 2``)
        n_freq (int, optional): number of filter banks from stft. (Default: ``201``)
        fixed_rate (float or None, optional): rate to speed up or slow down by.
            If None is provided, rate must be passed to the forward method. (Default: ``None``)

    .. note::

       The expected input is raw, complex-valued spectrogram.

    Example
        >>> spectrogram = torchaudio.transforms.Spectrogram(power=None)
        >>> stretch = torchaudio.transforms.TimeStretch()
        >>>
        >>> original = spectrogram(waveform)
        >>> stretched_1_2 = stretch(original, 1.2)
        >>> stretched_0_9 = stretch(original, 0.9)

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/specaugment_time_stretch.png
           :width: 600
           :alt: The visualization of stretched spectrograms.
    """
    __constants__ = ['fixed_rate']

    def __init__(self, hop_length: Optional[int]=None, n_freq: int=201, fixed_rate: Optional[float]=None) -> None:
        super(TimeStretch, self).__init__()
        self.fixed_rate = fixed_rate
        n_fft = (n_freq - 1) * 2
        hop_length = hop_length if hop_length is not None else n_fft // 2
        self.register_buffer('phase_advance', torch.linspace(0, math.pi * hop_length, n_freq)[..., None])

    def forward(self, complex_specgrams: Tensor, overriding_rate: Optional[float]=None) -> Tensor:
        """
        Args:
            complex_specgrams (Tensor):
                A tensor of dimension `(..., freq, num_frame)` with complex dtype.
            overriding_rate (float or None, optional): speed up to apply to this batch.
                If no rate is passed, use ``self.fixed_rate``. (Default: ``None``)

        Returns:
            Tensor:
                Stretched spectrogram. The resulting tensor is of the corresponding complex dtype
                as the input spectrogram, and the number of frames is changed to ``ceil(num_frame / rate)``.
        """
        if not torch.is_complex(complex_specgrams):
            warnings.warn('The input to TimeStretch must be complex type. Providing non-complex tensor produces invalid results.', stacklevel=4)
        if overriding_rate is None:
            if self.fixed_rate is None:
                raise ValueError('If no fixed_rate is specified, must pass a valid rate to the forward method.')
            rate = self.fixed_rate
        else:
            rate = overriding_rate
        return F.phase_vocoder(complex_specgrams, rate, self.phase_advance)