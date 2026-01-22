from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
class BeatTrackingProcessor(Processor):
    """
    Track the beats according to previously determined (local) tempo by
    iteratively aligning them around the estimated position [1]_.

    Parameters
    ----------
    look_aside : float, optional
        Look this fraction of the estimated beat interval to each side of the
        assumed next beat position to look for the most likely position of the
        next beat.
    look_ahead : float, optional
        Look `look_ahead` seconds in both directions to determine the local
        tempo and align the beats accordingly.
    tempo_estimator : :class:`TempoEstimationProcessor`, optional
        Use this processor to estimate the (local) tempo. If 'None' a default
        tempo estimator will be created and used.
    fps : float, optional
        Frames per second.
    kwargs : dict, optional
        Keyword arguments passed to
        :class:`madmom.features.tempo.TempoEstimationProcessor` if no
        `tempo_estimator` was given.

    Notes
    -----
    If `look_ahead` is not set, a constant tempo throughout the whole piece
    is assumed. If `look_ahead` is set, the local tempo (in a range +/-
    `look_ahead` seconds around the actual position) is estimated and then
    the next beat is tracked accordingly. This procedure is repeated from
    the new position to the end of the piece.

    Instead of the auto-correlation based method for tempo estimation proposed
    in [1]_, it uses a comb filter based method [2]_ per default. The behaviour
    can be controlled with the `tempo_method` parameter.

    References
    ----------
    .. [1] Sebastian Böck and Markus Schedl,
           "Enhanced Beat Tracking with Context-Aware Neural Networks",
           Proceedings of the 14th International Conference on Digital Audio
           Effects (DAFx), 2011.
    .. [2] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Accurate Tempo Estimation based on Recurrent Neural Networks and
           Resonating Comb Filters",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    Examples
    --------
    Create a BeatTrackingProcessor. The returned array represents the positions
    of the beats in seconds, thus the expected sampling rate has to be given.

    >>> proc = BeatTrackingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.BeatTrackingProcessor object at 0x...>

    Call this BeatTrackingProcessor with the beat activation function returned
    by RNNBeatProcessor to obtain the beat positions.

    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)
    array([0.11, 0.45, 0.79, 1.13, 1.47, 1.81, 2.15, 2.49])

    """
    LOOK_ASIDE = 0.2
    LOOK_AHEAD = 10.0

    def __init__(self, look_aside=LOOK_ASIDE, look_ahead=LOOK_AHEAD, fps=None, tempo_estimator=None, **kwargs):
        self.look_aside = look_aside
        self.look_ahead = look_ahead
        self.fps = fps
        if tempo_estimator is None:
            from .tempo import TempoEstimationProcessor
            tempo_estimator = TempoEstimationProcessor(fps=fps, **kwargs)
        self.tempo_estimator = tempo_estimator

    def process(self, activations, **kwargs):
        """
        Detect the beats in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        beats : numpy array
            Detected beat positions [seconds].

        """
        act_smooth = int(self.fps * self.tempo_estimator.act_smooth)
        activations = smooth_signal(activations, act_smooth)
        if self.look_ahead is None:
            histogram = self.tempo_estimator.interval_histogram(activations)
            interval = self.tempo_estimator.dominant_interval(histogram)
            detections = detect_beats(activations, interval, self.look_aside)
        else:
            look_ahead_frames = int(self.look_ahead * self.fps)
            detections = []
            pos = 0
            while pos < len(activations):
                act = signal_frame(activations, pos, look_ahead_frames * 2, 1)
                histogram = self.tempo_estimator.interval_histogram(act)
                interval = self.tempo_estimator.dominant_interval(histogram)
                positions = detect_beats(act, interval, self.look_aside)
                positions += pos - look_ahead_frames
                next_pos = detections[-1] + self.tempo_estimator.min_interval if detections else 0
                positions = positions[positions >= next_pos]
                pos = positions[np.abs(positions - pos).argmin()]
                detections.append(pos)
                pos += interval
        detections = np.array(detections) / float(self.fps)
        return detections[np.searchsorted(detections, 0):]

    @staticmethod
    def add_arguments(parser, look_aside=LOOK_ASIDE, look_ahead=LOOK_AHEAD):
        """
        Add beat tracking related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        look_aside : float, optional
            Look this fraction of the estimated beat interval to each side of
            the assumed next beat position to look for the most likely position
            of the next beat.
        look_ahead : float, optional
            Look `look_ahead` seconds in both directions to determine the local
            tempo and align the beats accordingly.

        Returns
        -------
        parser_group : argparse argument group
            Beat tracking argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        g = parser.add_argument_group('beat detection arguments')
        if look_aside is not None:
            g.add_argument('--look_aside', action='store', type=float, default=look_aside, help='look this fraction of a beat interval to each side of the assumed next beat position to look for the most likely position of the next beat [default=%(default).2f]')
        if look_ahead is not None:
            g.add_argument('--look_ahead', action='store', type=float, default=look_ahead, help='look this many seconds in both directions to determine the local tempo and align the beats accordingly [default=%(default).2f]')
        return g