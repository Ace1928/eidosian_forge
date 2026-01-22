import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class TimeMasking(_AxisMasking):
    """Apply masking to a spectrogram in the time domain.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Proposed in *SpecAugment* :cite:`specaugment`.

    Args:
        time_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, time_mask_param).
        iid_masks (bool, optional): whether to apply different masks to each
            example/channel in the batch. (Default: ``False``)
            This option is applicable only when the input tensor >= 3D.
        p (float, optional): maximum proportion of time steps that can be masked.
            Must be within range [0.0, 1.0]. (Default: 1.0)

    Example
        >>> spectrogram = torchaudio.transforms.Spectrogram()
        >>> masking = torchaudio.transforms.TimeMasking(time_mask_param=80)
        >>>
        >>> original = spectrogram(waveform)
        >>> masked = masking(original)

        .. image::  https://download.pytorch.org/torchaudio/doc-assets/specaugment_time_masking1.png
           :alt: The original spectrogram

        .. image::  https://download.pytorch.org/torchaudio/doc-assets/specaugment_time_masking2.png
           :alt: The spectrogram masked along time axis
    """

    def __init__(self, time_mask_param: int, iid_masks: bool=False, p: float=1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'The value of p must be between 0.0 and 1.0 ({p} given).')
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks, p=p)