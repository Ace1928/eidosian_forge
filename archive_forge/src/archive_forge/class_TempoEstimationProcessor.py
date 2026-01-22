from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
class TempoEstimationProcessor(OnlineProcessor):
    """
    Tempo Estimation Processor class.

    Parameters
    ----------
    method : {'comb', 'acf', 'dbn'}
        Method used for tempo estimation.
    min_bpm : float, optional
        Minimum tempo to detect [bpm].
    max_bpm : float, optional
        Maximum tempo to detect [bpm].
    act_smooth : float, optional (default: 0.14)
        Smooth the activation function over `act_smooth` seconds.
    hist_smooth : int, optional (default: 7)
        Smooth the tempo histogram over `hist_smooth` bins.
    alpha : float, optional
        Scaling factor for the comb filter.
    fps : float, optional
        Frames per second.
    histogram_processor : :class:`TempoHistogramProcessor`, optional
        Processor used to create a tempo histogram. If 'None', a default
        combfilter histogram processor will be created and used.
    kwargs : dict, optional
        Keyword arguments passed to :class:`CombFilterTempoHistogramProcessor`
        if no `histogram_processor` was given.

    Examples
    --------
    Create a TempoEstimationProcessor. The returned array represents the
    estimated tempi (given in beats per minute) and their relative strength.

    >>> proc = TempoEstimationProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.tempo.TempoEstimationProcessor object at 0x...>

    Call this TempoEstimationProcessor with the beat activation function
    obtained by RNNBeatProcessor to estimate the tempi.

    >>> from madmom.features.beats import RNNBeatProcessor
    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +NORMALIZE_WHITESPACE
    array([[176.47059,  0.47469],
           [117.64706,  0.17667],
           [240.     ,  0.15371],
           [ 68.96552,  0.09864],
           [ 82.19178,  0.09629]])

    """

    def __init__(self, method=METHOD, min_bpm=MIN_BPM, max_bpm=MAX_BPM, act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH, fps=None, online=False, histogram_processor=None, **kwargs):
        super(TempoEstimationProcessor, self).__init__(online=online)
        self.method = method
        self.act_smooth = act_smooth
        self.hist_smooth = hist_smooth
        self.fps = fps
        if self.online:
            self.visualize = kwargs.get('verbose', False)
        if histogram_processor is None:
            if method == 'acf':
                histogram_processor = ACFTempoHistogramProcessor
            elif method == 'comb':
                histogram_processor = CombFilterTempoHistogramProcessor
            elif method == 'dbn':
                histogram_processor = DBNTempoHistogramProcessor
                self.act_smooth = None
            else:
                raise ValueError('tempo histogram method unknown.')
            histogram_processor = histogram_processor(min_bpm=min_bpm, max_bpm=max_bpm, fps=fps, online=online, **kwargs)
        self.histogram_processor = histogram_processor

    @property
    def min_bpm(self):
        """Minimum tempo [bpm]."""
        return self.histogram_processor.min_bpm

    @property
    def max_bpm(self):
        """Maximum  tempo [bpm]."""
        return self.histogram_processor.max_bpm

    @property
    def intervals(self):
        """Beat intervals [frames]."""
        return self.histogram_processor.intervals

    @property
    def min_interval(self):
        """Minimum beat interval [frames]."""
        return self.histogram_processor.min_interval

    @property
    def max_interval(self):
        """Maximum beat interval [frames]."""
        return self.histogram_processor.max_interval

    def reset(self):
        """Reset to initial state."""
        self.histogram_processor.reset()

    def process_offline(self, activations, **kwargs):
        """
        Detect the tempi from the (beat) activations.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        tempi : numpy array
            Array with the dominant tempi [bpm] (first column) and their
            relative strengths (second column).

        """
        if self.act_smooth is not None:
            act_smooth = int(round(self.fps * self.act_smooth))
            activations = smooth_signal(activations, act_smooth)
        histogram = self.interval_histogram(activations.astype(np.float))
        histogram = smooth_histogram(histogram, self.hist_smooth)
        return detect_tempo(histogram, self.fps)

    def process_online(self, activations, reset=True, **kwargs):
        """
        Detect the tempi from the (beat) activations in online mode.

        Parameters
        ----------
        activations : numpy array
            Beat activation function processed frame by frame.
        reset : bool, optional
            Reset the TempoEstimationProcessor to its initial state before
            processing.

        Returns
        -------
        tempi : numpy array
            Array with the dominant tempi [bpm] (first column) and their
            relative strengths (second column).

        """
        histogram = self.interval_histogram(activations, reset=reset)
        histogram = smooth_histogram(histogram, self.hist_smooth)
        tempo = detect_tempo(histogram, self.fps)
        if self.visualize:
            display = ''
            for i, display_tempo in enumerate(tempo[:3], start=1):
                display += '| ' + str(round(display_tempo[0], 1)) + ' '
                display += min(int(display_tempo[1] * 50), 18) * '*'
                display = display.ljust(i * 26)
            sys.stderr.write('\r%s' % ''.join(display) + '|')
            sys.stderr.flush()
        return tempo

    def interval_histogram(self, activations, **kwargs):
        """
        Compute the histogram of the beat intervals.

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
        return self.histogram_processor(activations, **kwargs)

    def dominant_interval(self, histogram):
        """
        Extract the dominant interval of the given histogram.

        Parameters
        ----------
        histogram : tuple
            Histogram (tuple of 2 numpy arrays, the first giving the strengths
            of the bins and the second corresponding delay values).

        Returns
        -------
        interval : int
            Dominant interval.

        """
        return dominant_interval(histogram, self.hist_smooth)

    @staticmethod
    def add_arguments(parser, method=None, min_bpm=None, max_bpm=None, act_smooth=None, hist_smooth=None, hist_buffer=None, alpha=None):
        """
        Add tempo estimation related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser.
        method : {'comb', 'acf', 'dbn'}
            Method used for tempo estimation.
        min_bpm : float, optional
            Minimum tempo to detect [bpm].
        max_bpm : float, optional
            Maximum tempo to detect [bpm].
        act_smooth : float, optional
            Smooth the activation function over `act_smooth` seconds.
        hist_smooth : int, optional
            Smooth the tempo histogram over `hist_smooth` bins.
        hist_buffer : float, optional
            Aggregate the tempo histogram over `hist_buffer` seconds.
        alpha : float, optional
            Scaling factor for the comb filter.

        Returns
        -------
        parser_group : argparse argument group
            Tempo argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        g = parser.add_argument_group('tempo estimation arguments')
        if method is not None:
            g.add_argument('--method', action='store', type=str, default=method, choices=['acf', 'comb', 'dbn'], help='which method to use [default=%(default)s]')
        if min_bpm is not None:
            g.add_argument('--min_bpm', action='store', type=float, default=min_bpm, help='minimum tempo [bpm, default=%(default).2f]')
        if max_bpm is not None:
            g.add_argument('--max_bpm', action='store', type=float, default=max_bpm, help='maximum tempo [bpm, default=%(default).2f]')
        if act_smooth is not None:
            g.add_argument('--act_smooth', action='store', type=float, default=act_smooth, help='smooth the activations over N seconds [default=%(default).2f]')
        if hist_smooth is not None:
            g.add_argument('--hist_smooth', action='store', type=int, default=hist_smooth, help='smooth the tempo histogram over N bins [default=%(default)d]')
        if hist_buffer is not None:
            g.add_argument('--hist_buffer', action='store', type=float, default=hist_buffer, help='aggregate the tempo histogram over N seconds [default=%(default).2f]')
        if alpha is not None:
            g.add_argument('--alpha', action='store', type=float, default=alpha, help='alpha for comb filter tempo estimation [default=%(default).2f]')
        return g