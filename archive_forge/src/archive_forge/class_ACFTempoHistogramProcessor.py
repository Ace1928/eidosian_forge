from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
class ACFTempoHistogramProcessor(TempoHistogramProcessor):
    """
    Create a tempo histogram with autocorrelation.

    Parameters
    ----------
    min_bpm : float, optional
        Minimum tempo to detect [bpm].
    max_bpm : float, optional
        Maximum tempo to detect [bpm].
    hist_buffer : float
        Aggregate the tempo histogram over `hist_buffer` seconds.
    fps : float, optional
        Frames per second.
    online : bool, optional
        Operate in online (i.e. causal) mode.

    """

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, hist_buffer=HIST_BUFFER, fps=None, online=False, **kwargs):
        super(ACFTempoHistogramProcessor, self).__init__(min_bpm=min_bpm, max_bpm=max_bpm, hist_buffer=hist_buffer, fps=fps, online=online, **kwargs)
        if self.online:
            self._act_buffer = BufferProcessor((self.max_interval + 1, 1))

    def reset(self):
        """Reset to initial state."""
        super(ACFTempoHistogramProcessor, self).reset()
        self._act_buffer.reset()

    def process_offline(self, activations, **kwargs):
        """
        Compute the histogram of the beat intervals with the autocorrelation
        function.

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
        return interval_histogram_acf(activations, self.min_interval, self.max_interval)

    def process_online(self, activations, reset=True, **kwargs):
        """
        Compute the histogram of the beat intervals with the autocorrelation
        function in online mode.

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
        for act in activations:
            bins = act * self._act_buffer[-self.intervals].T
            self._act_buffer(act)
            bins = self._hist_buffer(bins)
        return (np.sum(bins, axis=0), self.intervals)