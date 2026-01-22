import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def _lfilter_core(waveform: Tensor, a_coeffs: Tensor, b_coeffs: Tensor) -> Tensor:
    if a_coeffs.size() != b_coeffs.size():
        raise ValueError(f'Expected coeffs to be the same size.Found a_coeffs size: {a_coeffs.size()}, b_coeffs size: {b_coeffs.size()}')
    if waveform.ndim != 3:
        raise ValueError(f'Expected waveform to be 3 dimensional. Found: {waveform.ndim}')
    if not waveform.device == a_coeffs.device == b_coeffs.device:
        raise ValueError(f'Expected waveform and coeffs to be on the same device.Found: waveform device:{waveform.device}, a_coeffs device: {a_coeffs.device}, b_coeffs device: {b_coeffs.device}')
    n_batch, n_channel, n_sample = waveform.size()
    n_order = a_coeffs.size(1)
    if n_order <= 0:
        raise ValueError(f'Expected n_order to be positive. Found: {n_order}')
    padded_waveform = torch.nn.functional.pad(waveform, [n_order - 1, 0])
    padded_output_waveform = torch.zeros_like(padded_waveform)
    a_coeffs_flipped = a_coeffs.flip(1)
    b_coeffs_flipped = b_coeffs.flip(1)
    input_signal_windows = torch.nn.functional.conv1d(padded_waveform, b_coeffs_flipped.unsqueeze(1), groups=n_channel)
    input_signal_windows.div_(a_coeffs[:, :1])
    a_coeffs_flipped.div_(a_coeffs[:, :1])
    if input_signal_windows.device == torch.device('cpu') and a_coeffs_flipped.device == torch.device('cpu') and (padded_output_waveform.device == torch.device('cpu')):
        _lfilter_core_cpu_loop(input_signal_windows, a_coeffs_flipped, padded_output_waveform)
    else:
        _lfilter_core_generic_loop(input_signal_windows, a_coeffs_flipped, padded_output_waveform)
    output = padded_output_waveform[:, :, n_order - 1:]
    return output