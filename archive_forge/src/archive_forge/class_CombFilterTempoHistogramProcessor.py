from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
class CombFilterTempoHistogramProcessor(TempoHistogramProcessor):
    """
    Create a tempo histogram with a bank of resonating comb filters.

    Parameters
    ----------
    min_bpm : float, optional
        Minimum tempo to detect [bpm].
    max_bpm : float, optional
        Maximum tempo to detect [bpm].
    alpha : float, optional
        Scaling factor for the comb filter.
    hist_buffer : float
        Aggregate the tempo histogram over `hist_buffer` seconds.
    fps : float, optional
        Frames per second.
    online : bool, optional
        Operate in online (i.e. causal) mode.

    """

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, alpha=ALPHA, hist_buffer=HIST_BUFFER, fps=None, online=False, **kwargs):
        super(CombFilterTempoHistogramProcessor, self).__init__(min_bpm=min_bpm, max_bpm=max_bpm, hist_buffer=hist_buffer, fps=fps, online=online, **kwargs)
        self.alpha = alpha
        if self.online:
            self._comb_buffer = BufferProcessor((self.max_interval + 1, len(self.intervals)))

    def reset(self):
        """Reset to initial state."""
        super(CombFilterTempoHistogramProcessor, self).reset()
        self._comb_buffer.reset()

    def process_offline(self, activations, **kwargs):
        """
        Compute the histogram of the beat intervals with a bank of resonating
        comb filters.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        histogram_bins : numpy array
            Bins of the beat interval histogram.
        histogram_delays : numpy array
            Corresponding delays [frames].

        """
        return interval_histogram_comb(activations, self.alpha, self.min_interval, self.max_interval)

    def process_online(self, activations, reset=True, **kwargs):
        """
        Compute the histogram of the beat intervals with a bank of resonating
        comb filters in online mode.

        Parameters
        ----------
        activations : numpy float
            Beat activation function.
        reset : bool, optional
            Reset to initial state before processing.

        Returns
        -------
        histogram_bins : numpy array
            Bins of the tempo histogram.
        histogram_delays : numpy array
            Corresponding delays [frames].

        """
        if reset:
            self.reset()
        idx = [-self.intervals, np.arange(len(self.intervals))]
        for act in activations:
            y_n = act + self.alpha * self._comb_buffer[idx]
            self._comb_buffer(y_n)
            act_max = y_n == np.max(y_n, axis=-1)[..., np.newaxis]
            bins = y_n * act_max
            bins = self._hist_buffer(bins)
        return (np.sum(bins, axis=0), self.intervals)