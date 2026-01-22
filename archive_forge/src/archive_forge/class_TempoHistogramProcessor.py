from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
class TempoHistogramProcessor(OnlineProcessor):
    """
    Tempo Histogram Processor class.

    Parameters
    ----------
    min_bpm : float
        Minimum tempo to detect [bpm].
    max_bpm : float
        Maximum tempo to detect [bpm].
    hist_buffer : float
        Aggregate the tempo histogram over `hist_buffer` seconds.
    fps : float, optional
        Frames per second.

    Notes
    -----
    This abstract class provides the basic tempo histogram functionality.
    Please use one of the following implementations:

    - :class:`CombFilterTempoHistogramProcessor`,
    - :class:`ACFTempoHistogramProcessor` or
    - :class:`DBNTempoHistogramProcessor`.

    """

    def __init__(self, min_bpm, max_bpm, hist_buffer=HIST_BUFFER, fps=None, online=False, **kwargs):
        super(TempoHistogramProcessor, self).__init__(online=online)
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.hist_buffer = hist_buffer
        self.fps = fps
        if self.online:
            self._hist_buffer = BufferProcessor((int(hist_buffer * self.fps), len(self.intervals)))

    @property
    def min_interval(self):
        """Minimum beat interval [frames]."""
        return int(np.floor(60.0 * self.fps / self.max_bpm))

    @property
    def max_interval(self):
        """Maximum beat interval [frames]."""
        return int(np.ceil(60.0 * self.fps / self.min_bpm))

    @property
    def intervals(self):
        """Beat intervals [frames]."""
        return np.arange(self.min_interval, self.max_interval + 1)

    def reset(self):
        """Reset the tempo histogram aggregation buffer."""
        self._hist_buffer.reset()