import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class FrequencyMasking(_AxisMasking):
    """Apply masking to a spectrogram in the frequency domain.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Proposed in *SpecAugment* :cite:`specaugment`.

    Args:
        freq_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, freq_mask_param).
        iid_masks (bool, optional): whether to apply different masks to each
            example/channel in the batch. (Default: ``False``)
            This option is applicable only when the input tensor >= 3D.

    Example
        >>> spectrogram = torchaudio.transforms.Spectrogram()
        >>> masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=80)
        >>>
        >>> original = spectrogram(waveform)
        >>> masked = masking(original)

        .. image::  https://download.pytorch.org/torchaudio/doc-assets/specaugment_freq_masking1.png
           :alt: The original spectrogram

        .. image::  https://download.pytorch.org/torchaudio/doc-assets/specaugment_freq_masking2.png
           :alt: The spectrogram masked along frequency axis
    """

    def __init__(self, freq_mask_param: int, iid_masks: bool=False) -> None:
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks)