import warnings
from sys import platform
from typing import Optional
import torch
import torchaudio
Plays audio through specified or available output device.

    .. warning::
       This function is currently only supported on MacOS, and requires
       libavdevice (FFmpeg) with ``audiotoolbox`` output device.

    .. note::
       This function can play up to two audio channels.

    Args:
        waveform: Tensor containing the audio to play.
            Expected shape: `(time, num_channels)`.
        sample_rate: Sample rate of the audio to play.
        device: Output device to use. If None, the default device is used.
    