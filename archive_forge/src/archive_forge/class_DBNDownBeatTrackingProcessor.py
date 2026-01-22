from __future__ import absolute_import, division, print_function
import sys
import warnings
import numpy as np
from .beats_hmm import (BarStateSpace, BarTransitionModel,
from ..ml.hmm import HiddenMarkovModel
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import string_types
class DBNDownBeatTrackingProcessor(Processor):
    """
    Downbeat tracking with RNNs and a dynamic Bayesian network (DBN)
    approximated by a Hidden Markov Model (HMM).

    Parameters
    ----------
    beats_per_bar : int or list
        Number of beats per bar to be modeled. Can be either a single number
        or a list or array with bar lengths (in beats).
    min_bpm : float or list, optional
        Minimum tempo used for beat tracking [bpm]. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    max_bpm : float or list, optional
        Maximum tempo used for beat tracking [bpm]. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    num_tempi : int or list, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacing, otherwise a linear spacing. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    transition_lambda : float or list, optional
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).  If a list is
        given, each item corresponds to the number of beats per bar at the
        same position.
    observation_lambda : int, optional
        Split one (down-)beat period into `observation_lambda` parts, the first
        representing (down-)beat states and the remaining non-beat states.
    threshold : float, optional
        Threshold the RNN (down-)beat activations before Viterbi decoding.
    correct : bool, optional
        Correct the beats (i.e. align them to the nearest peak of the
        (down-)beat activation function).
    fps : float, optional
        Frames per second.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    Create a DBNDownBeatTrackingProcessor. The returned array represents the
    positions of the beats and their position inside the bar. The position is
    given in seconds, thus the expected sampling rate is needed. The position
    inside the bar follows the natural counting and starts at 1.

    The number of beats per bar which should be modelled must be given, all
    other parameters (e.g. tempo range) are optional but must have the same
    length as `beats_per_bar`, i.e. must be given for each bar length.

    >>> proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.downbeats.DBNDownBeatTrackingProcessor object at 0x...>

    Call this DBNDownBeatTrackingProcessor with the beat activation function
    returned by RNNDownBeatProcessor to obtain the beat positions.

    >>> act = RNNDownBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[0.09, 1. ],
           [0.45, 2. ],
           ...,
           [2.14, 3. ],
           [2.49, 4. ]])

    """
    MIN_BPM = 55.0
    MAX_BPM = 215.0
    NUM_TEMPI = 60
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    THRESHOLD = 0.05
    CORRECT = True

    def __init__(self, beats_per_bar, min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA, observation_lambda=OBSERVATION_LAMBDA, threshold=THRESHOLD, correct=CORRECT, fps=None, **kwargs):
        beats_per_bar = np.array(beats_per_bar, ndmin=1)
        min_bpm = np.array(min_bpm, ndmin=1)
        max_bpm = np.array(max_bpm, ndmin=1)
        num_tempi = np.array(num_tempi, ndmin=1)
        transition_lambda = np.array(transition_lambda, ndmin=1)
        if len(min_bpm) != len(beats_per_bar):
            min_bpm = np.repeat(min_bpm, len(beats_per_bar))
        if len(max_bpm) != len(beats_per_bar):
            max_bpm = np.repeat(max_bpm, len(beats_per_bar))
        if len(num_tempi) != len(beats_per_bar):
            num_tempi = np.repeat(num_tempi, len(beats_per_bar))
        if len(transition_lambda) != len(beats_per_bar):
            transition_lambda = np.repeat(transition_lambda, len(beats_per_bar))
        if not len(min_bpm) == len(max_bpm) == len(num_tempi) == len(beats_per_bar) == len(transition_lambda):
            raise ValueError('`min_bpm`, `max_bpm`, `num_tempi`, `num_beats` and `transition_lambda` must all have the same length.')
        num_threads = min(len(beats_per_bar), kwargs.get('num_threads', 1))
        self.map = map
        if num_threads != 1:
            import multiprocessing as mp
            self.map = mp.Pool(num_threads).map
        min_interval = 60.0 * fps / max_bpm
        max_interval = 60.0 * fps / min_bpm
        self.hmms = []
        for b, beats in enumerate(beats_per_bar):
            st = BarStateSpace(beats, min_interval[b], max_interval[b], num_tempi[b])
            tm = BarTransitionModel(st, transition_lambda[b])
            om = RNNDownBeatTrackingObservationModel(st, observation_lambda)
            self.hmms.append(HiddenMarkovModel(tm, om))
        self.beats_per_bar = beats_per_bar
        self.threshold = threshold
        self.correct = correct
        self.fps = fps

    def process(self, activations, **kwargs):
        """
        Detect the (down-)beats in the given activation function.

        Parameters
        ----------
        activations : numpy array, shape (num_frames, 2)
            Activation function with probabilities corresponding to beats
            and downbeats given in the first and second column, respectively.

        Returns
        -------
        beats : numpy array, shape (num_beats, 2)
            Detected (down-)beat positions [seconds] and beat numbers.

        """
        import itertools as it
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        if not activations.any():
            return np.empty((0, 2))
        results = list(self.map(_process_dbn, zip(self.hmms, it.repeat(activations))))
        best = np.argmax(np.asarray(results)[:, 1])
        path, _ = results[best]
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        positions = st.state_positions[path]
        beat_numbers = positions.astype(int) + 1
        if self.correct:
            beats = np.empty(0, dtype=np.int)
            beat_range = om.pointers[path] >= 1
            idx = np.nonzero(np.diff(beat_range.astype(np.int)))[0] + 1
            if beat_range[0]:
                idx = np.r_[0, idx]
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1
        return np.vstack(((beats + first) / float(self.fps), beat_numbers[beats])).T

    @staticmethod
    def add_arguments(parser, beats_per_bar, min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA, observation_lambda=OBSERVATION_LAMBDA, threshold=THRESHOLD, correct=CORRECT):
        """
        Add DBN downbeat tracking related arguments to an existing parser
        object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        beats_per_bar : int or list, optional
            Number of beats per bar to be modeled. Can be either a single
            number or a list with bar lengths (in beats).
        min_bpm : float or list, optional
            Minimum tempo used for beat tracking [bpm]. If a list is given,
            each item corresponds to the number of beats per bar at the same
            position.
        max_bpm : float or list, optional
            Maximum tempo used for beat tracking [bpm]. If a list is given,
            each item corresponds to the number of beats per bar at the same
            position.
        num_tempi : int or list, optional
            Number of tempi to model; if set, limit the number of tempi and use
            a log spacing, otherwise a linear spacing. If a list is given,
            each item corresponds to the number of beats per bar at the same
            position.
        transition_lambda : float or list, optional
            Lambda for the exponential tempo change distribution (higher values
            prefer a constant tempo over a tempo change from one beat to the
            next one). If a list is given, each item corresponds to the number
            of beats per bar at the same position.
        observation_lambda : float, optional
            Split one (down-)beat period into `observation_lambda` parts, the
            first representing (down-)beat states and the remaining non-beat
            states.
        threshold : float, optional
            Threshold the RNN (down-)beat activations before Viterbi decoding.
        correct : bool, optional
            Correct the beats (i.e. align them to the nearest peak of the
            (down-)beat activation function).

        Returns
        -------
        parser_group : argparse argument group
            DBN downbeat tracking argument parser group

        """
        from ..utils import OverrideDefaultListAction
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        g.add_argument('--beats_per_bar', action=OverrideDefaultListAction, default=beats_per_bar, type=int, sep=',', help='number of beats per bar to be modeled (comma separated list of bar length in beats) [default=%(default)s]')
        g.add_argument('--min_bpm', action=OverrideDefaultListAction, default=min_bpm, type=float, sep=',', help='minimum tempo (comma separated list with one value per bar length) [bpm, default=%(default)s]')
        g.add_argument('--max_bpm', action=OverrideDefaultListAction, default=max_bpm, type=float, sep=',', help='maximum tempo (comma separated list with one value per bar length) [bpm, default=%(default)s]')
        g.add_argument('--num_tempi', action=OverrideDefaultListAction, default=num_tempi, type=int, sep=',', help='limit the number of tempi; if set, align the tempi with log spacings, otherwise linearly (comma separated list with one value per bar length) [default=%(default)s]')
        g.add_argument('--transition_lambda', action=OverrideDefaultListAction, default=transition_lambda, type=float, sep=',', help='lambda of the tempo transition distribution; higher values prefer a constant tempo over a tempo change from one beat to the next one (comma separated list with one value per bar length) [default=%(default)s]')
        g.add_argument('--observation_lambda', action='store', type=float, default=observation_lambda, help='split one (down-)beat period into N parts, the first representing beat states and the remaining non-beat states [default=%(default)i]')
        g.add_argument('-t', dest='threshold', action='store', type=float, default=threshold, help='threshold the observations before Viterbi decoding [default=%(default).2f]')
        if correct is True:
            g.add_argument('--no_correct', dest='correct', action='store_false', default=correct, help='do not correct the (down-)beat positions (i.e. do not align them to the nearest peak of the (down-)beat activation function)')
        elif correct is False:
            g.add_argument('--correct', dest='correct', action='store_true', default=correct, help='correct the (down-)beat positions (i.e. align them to the nearest peak of the (down-)beat  activation function)')
        g = parser.add_argument_group('output arguments')
        g.add_argument('--downbeats', action='store_true', default=False, help='output only the downbeats')
        return g