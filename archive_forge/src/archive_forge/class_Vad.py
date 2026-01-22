import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class Vad(torch.nn.Module):
    """Voice Activity Detector. Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Attempts to trim silence and quiet background sounds from the ends of recordings of speech.
    The algorithm currently uses a simple cepstral power measurement to detect voice,
    so may be fooled by other things, especially music.

    The effect can trim only from the front of the audio,
    so in order to trim from the back, the reverse effect must also be used.

    Args:
        sample_rate (int): Sample rate of audio signal.
        trigger_level (float, optional): The measurement level used to trigger activity detection.
            This may need to be changed depending on the noise level, signal level,
            and other characteristics of the input audio. (Default: 7.0)
        trigger_time (float, optional): The time constant (in seconds)
            used to help ignore short bursts of sound. (Default: 0.25)
        search_time (float, optional): The amount of audio (in seconds)
            to search for quieter/shorter bursts of audio to include prior
            to the detected trigger point. (Default: 1.0)
        allowed_gap (float, optional): The allowed gap (in seconds) between
            quiteter/shorter bursts of audio to include prior
            to the detected trigger point. (Default: 0.25)
        pre_trigger_time (float, optional): The amount of audio (in seconds) to preserve
            before the trigger point and any found quieter/shorter bursts. (Default: 0.0)
        boot_time (float, optional) The algorithm (internally) uses adaptive noise
            estimation/reduction in order to detect the start of the wanted audio.
            This option sets the time for the initial noise estimate. (Default: 0.35)
        noise_up_time (float, optional) Time constant used by the adaptive noise estimator
            for when the noise level is increasing. (Default: 0.1)
        noise_down_time (float, optional) Time constant used by the adaptive noise estimator
            for when the noise level is decreasing. (Default: 0.01)
        noise_reduction_amount (float, optional) Amount of noise reduction to use in
            the detection algorithm (e.g. 0, 0.5, ...). (Default: 1.35)
        measure_freq (float, optional) Frequency of the algorithm’s
            processing/measurements. (Default: 20.0)
        measure_duration: (float or None, optional) Measurement duration.
            (Default: Twice the measurement period; i.e. with overlap.)
        measure_smooth_time (float, optional) Time constant used to smooth
            spectral measurements. (Default: 0.4)
        hp_filter_freq (float, optional) "Brick-wall" frequency of high-pass filter applied
            at the input to the detector algorithm. (Default: 50.0)
        lp_filter_freq (float, optional) "Brick-wall" frequency of low-pass filter applied
            at the input to the detector algorithm. (Default: 6000.0)
        hp_lifter_freq (float, optional) "Brick-wall" frequency of high-pass lifter used
            in the detector algorithm. (Default: 150.0)
        lp_lifter_freq (float, optional) "Brick-wall" frequency of low-pass lifter used
            in the detector algorithm. (Default: 2000.0)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> waveform_reversed, sample_rate = apply_effects_tensor(waveform, sample_rate, [["reverse"]])
        >>> transform = transforms.Vad(sample_rate=sample_rate, trigger_level=7.5)
        >>> waveform_reversed_front_trim = transform(waveform_reversed)
        >>> waveform_end_trim, sample_rate = apply_effects_tensor(
        >>>     waveform_reversed_front_trim, sample_rate, [["reverse"]]
        >>> )

    Reference:
        - http://sox.sourceforge.net/sox.html
    """

    def __init__(self, sample_rate: int, trigger_level: float=7.0, trigger_time: float=0.25, search_time: float=1.0, allowed_gap: float=0.25, pre_trigger_time: float=0.0, boot_time: float=0.35, noise_up_time: float=0.1, noise_down_time: float=0.01, noise_reduction_amount: float=1.35, measure_freq: float=20.0, measure_duration: Optional[float]=None, measure_smooth_time: float=0.4, hp_filter_freq: float=50.0, lp_filter_freq: float=6000.0, hp_lifter_freq: float=150.0, lp_lifter_freq: float=2000.0) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.trigger_level = trigger_level
        self.trigger_time = trigger_time
        self.search_time = search_time
        self.allowed_gap = allowed_gap
        self.pre_trigger_time = pre_trigger_time
        self.boot_time = boot_time
        self.noise_up_time = noise_up_time
        self.noise_down_time = noise_down_time
        self.noise_reduction_amount = noise_reduction_amount
        self.measure_freq = measure_freq
        self.measure_duration = measure_duration
        self.measure_smooth_time = measure_smooth_time
        self.hp_filter_freq = hp_filter_freq
        self.lp_filter_freq = lp_filter_freq
        self.hp_lifter_freq = hp_lifter_freq
        self.lp_lifter_freq = lp_lifter_freq

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of dimension `(channels, time)` or `(time)`
                Tensor of shape `(channels, time)` is treated as a multi-channel recording
                of the same event and the resulting output will be trimmed to the earliest
                voice activity in any channel.
        """
        return F.vad(waveform=waveform, sample_rate=self.sample_rate, trigger_level=self.trigger_level, trigger_time=self.trigger_time, search_time=self.search_time, allowed_gap=self.allowed_gap, pre_trigger_time=self.pre_trigger_time, boot_time=self.boot_time, noise_up_time=self.noise_up_time, noise_down_time=self.noise_down_time, noise_reduction_amount=self.noise_reduction_amount, measure_freq=self.measure_freq, measure_duration=self.measure_duration, measure_smooth_time=self.measure_smooth_time, hp_filter_freq=self.hp_filter_freq, lp_filter_freq=self.lp_filter_freq, hp_lifter_freq=self.hp_lifter_freq, lp_lifter_freq=self.lp_lifter_freq)