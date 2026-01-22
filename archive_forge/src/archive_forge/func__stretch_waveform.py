import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated
from .filtering import highpass_biquad, treble_biquad
def _stretch_waveform(waveform: Tensor, n_steps: int, bins_per_octave: int=12, n_fft: int=512, win_length: Optional[int]=None, hop_length: Optional[int]=None, window: Optional[Tensor]=None) -> Tensor:
    """
    Pitch shift helper function to preprocess and stretch waveform before resampling step.

    Args:
        See pitch_shift arg descriptions.

    Returns:
        Tensor: The preprocessed waveform stretched prior to resampling.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(window_length=win_length, device=waveform.device)
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])
    ori_len = shape[-1]
    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    spec_f = torch.stft(input=waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    phase_advance = torch.linspace(0, math.pi * hop_length, spec_f.shape[-2], device=spec_f.device)[..., None]
    spec_stretch = phase_vocoder(spec_f, rate, phase_advance)
    len_stretch = int(round(ori_len / rate))
    waveform_stretch = torch.istft(spec_stretch, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=len_stretch)
    return waveform_stretch