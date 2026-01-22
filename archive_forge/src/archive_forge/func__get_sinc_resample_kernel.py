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
def _get_sinc_resample_kernel(orig_freq: int, new_freq: int, gcd: int, lowpass_filter_width: int=6, rolloff: float=0.99, resampling_method: str='sinc_interp_hann', beta: Optional[float]=None, device: torch.device=_CPU, dtype: Optional[torch.dtype]=None):
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception('Frequencies must be of integer type to ensure quality resampling computation. To work around this, manually convert both frequencies to integer values that maintain their resampling rate ratio before passing them into the function. Example: To downsample a 44100 hz waveform by a factor of 8, use `orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. For more information, please refer to https://github.com/pytorch/audio/issues/1487.')
    if resampling_method in ['sinc_interpolation', 'kaiser_window']:
        method_map = {'sinc_interpolation': 'sinc_interp_hann', 'kaiser_window': 'sinc_interp_kaiser'}
        warnings.warn(f'"{resampling_method}" resampling method name is being deprecated and replaced by "{method_map[resampling_method]}" in the next release. The default behavior remains unchanged.', stacklevel=3)
    elif resampling_method not in ['sinc_interp_hann', 'sinc_interp_kaiser']:
        raise ValueError('Invalid resampling method: {}'.format(resampling_method))
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    if lowpass_filter_width <= 0:
        raise ValueError('Low pass filter width should be positive.')
    base_freq = min(orig_freq, new_freq)
    base_freq *= rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx_dtype = dtype if dtype is not None else torch.float64
    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    if resampling_method == 'sinc_interp_hann':
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = torch.tensor(float(beta))
        window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)
    t *= math.pi
    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale
    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)
    return (kernels, width)