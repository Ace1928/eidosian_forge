import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
class Stretch2d(nn.Module):
    """Upscale the frequency and time dimensions of a spectrogram.

    Args:
        time_scale: the scale factor in time dimension
        freq_scale: the scale factor in frequency dimension

    Examples
        >>> stretch2d = Stretch2d(time_scale=10, freq_scale=5)

        >>> input = torch.rand(10, 100, 512)  # a random spectrogram
        >>> output = stretch2d(input)  # shape: (10, 500, 5120)
    """

    def __init__(self, time_scale: int, freq_scale: int) -> None:
        super().__init__()
        self.freq_scale = freq_scale
        self.time_scale = time_scale

    def forward(self, specgram: Tensor) -> Tensor:
        """Pass the input through the Stretch2d layer.

        Args:
            specgram (Tensor): the input sequence to the Stretch2d layer (..., n_freq, n_time).

        Return:
            Tensor shape: (..., n_freq * freq_scale, n_time * time_scale)
        """
        return specgram.repeat_interleave(self.freq_scale, -2).repeat_interleave(self.time_scale, -1)