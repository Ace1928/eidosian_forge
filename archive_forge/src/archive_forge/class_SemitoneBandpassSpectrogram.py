from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
class SemitoneBandpassSpectrogram(FilteredSpectrogram):
    """
    Construct a semitone spectrogram by using a time domain filterbank of
    bandpass filters as described in [1]_.

    Parameters
    ----------
    signal : Signal
        Signal instance.
    fps : float, optional
        Frame rate of the spectrogram [Hz].
    fmin : float, optional
        Lowest frequency of the spectrogram [Hz].
    fmax : float, optional
        Highest frequency of the spectrogram [Hz].

    References
    ----------
    .. [1] Meinard Müller,
           "Information retrieval for music and motion", Springer, 2007.

    """

    def __init__(self, signal, fps=50.0, fmin=27.5, fmax=4200.0):
        pass

    def __new__(cls, signal, fps=50.0, fmin=27.5, fmax=4200.0):
        from scipy.signal import filtfilt
        from .filters import SemitoneBandpassFilterbank
        from .signal import FramedSignal, Signal, resample
        if not isinstance(signal, Signal) or signal.num_channels != 1:
            signal = Signal(signal, num_channels=1)
        sample_rate = float(signal.sample_rate)
        signal_ = signal
        num_frames = np.round(len(signal) * fps / sample_rate) + 1
        filterbank = SemitoneBandpassFilterbank(fmin=fmin, fmax=fmax)
        bands = []
        for filt, band_sample_rate in zip(filterbank.filters, filterbank.band_sample_rates):
            frame_size = np.round(2 * band_sample_rate / float(fps))
            if band_sample_rate != signal.sample_rate:
                signal = resample(signal_, band_sample_rate)
            b, a = filt
            filtered_signal = filtfilt(b, a, signal)
            try:
                filtered_signal /= np.iinfo(signal.dtype).max
            except ValueError:
                pass
            filtered_signal = filtered_signal ** 2 / band_sample_rate * 22050.0
            frames = FramedSignal(filtered_signal, frame_size=frame_size, fps=fps, sample_rate=band_sample_rate, num_frames=num_frames)
            bands.append(np.sum(frames, axis=1))
        obj = np.vstack(bands).T.view(cls)
        obj.filterbank = filterbank
        obj.fps = fps
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.filterbank = getattr(obj, 'filterbank', None)
        self.fps = getattr(obj, 'fps', None)