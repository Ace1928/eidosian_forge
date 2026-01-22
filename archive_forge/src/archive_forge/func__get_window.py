import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
def _get_window(waveform: Tensor, padded_window_size: int, window_size: int, window_shift: int, window_type: str, blackman_coeff: float, snip_edges: bool, raw_energy: bool, energy_floor: float, dither: float, remove_dc_offset: bool, preemphasis_coefficient: float) -> Tuple[Tensor, Tensor]:
    """Gets a window and its log energy

    Returns:
        (Tensor, Tensor): strided_input of size (m, ``padded_window_size``) and signal_log_energy of size (m)
    """
    device, dtype = (waveform.device, waveform.dtype)
    epsilon = _get_epsilon(device, dtype)
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)
    if dither != 0.0:
        rand_gauss = torch.randn(strided_input.shape, device=device, dtype=dtype)
        strided_input = strided_input + rand_gauss * dither
    if remove_dc_offset:
        row_means = torch.mean(strided_input, dim=1).unsqueeze(1)
        strided_input = strided_input - row_means
    if raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)
    if preemphasis_coefficient != 0.0:
        offset_strided_input = torch.nn.functional.pad(strided_input.unsqueeze(0), (1, 0), mode='replicate').squeeze(0)
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]
    window_function = _feature_window_function(window_type, window_size, blackman_coeff, device, dtype).unsqueeze(0)
    strided_input = strided_input * window_function
    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = torch.nn.functional.pad(strided_input.unsqueeze(0), (0, padding_right), mode='constant', value=0).squeeze(0)
    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)
    return (strided_input, signal_log_energy)