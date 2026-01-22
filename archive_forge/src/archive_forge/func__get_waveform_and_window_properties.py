import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
def _get_waveform_and_window_properties(waveform: Tensor, channel: int, sample_frequency: float, frame_shift: float, frame_length: float, round_to_power_of_two: bool, preemphasis_coefficient: float) -> Tuple[Tensor, int, int, int]:
    """Gets the waveform and window properties"""
    channel = max(channel, 0)
    assert channel < waveform.size(0), 'Invalid channel {} for size {}'.format(channel, waveform.size(0))
    waveform = waveform[channel, :]
    window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size
    assert 2 <= window_size <= len(waveform), 'choose a window size {} that is [2, {}]'.format(window_size, len(waveform))
    assert 0 < window_shift, '`window_shift` must be greater than 0'
    assert padded_window_size % 2 == 0, 'the padded `window_size` must be divisible by two. use `round_to_power_of_two` or change `frame_length`'
    assert 0.0 <= preemphasis_coefficient <= 1.0, '`preemphasis_coefficient` must be between [0,1]'
    assert sample_frequency > 0, '`sample_frequency` must be greater than zero'
    return (waveform, window_shift, window_size, padded_window_size)