from __future__ import absolute_import, division, print_function
import sys
import warnings
import numpy as np
from .beats_hmm import (BarStateSpace, BarTransitionModel,
from ..ml.hmm import HiddenMarkovModel
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import string_types
class DBNBarTrackingProcessor(Processor):
    """
    Bar tracking with a dynamic Bayesian network (DBN) approximated by a
    Hidden Markov Model (HMM).

    Parameters
    ----------
    beats_per_bar : int or list
        Number of beats per bar to be modeled. Can be either a single number
        or a list or array with bar lengths (in beats).
    observation_weight : int, optional
        Weight for the downbeat activations.
    meter_change_prob : float, optional
        Probability to change meter at bar boundaries.

    Examples
    --------
    Create a DBNBarTrackingProcessor. The returned array represents the
    positions of the beats and their position inside the bar. The position
    inside the bar follows the natural counting and starts at 1.

    The number of beats per bar which should be modelled must be given, all
    other parameters (e.g. probability to change the meter at bar boundaries)
    are optional but must have the same length as `beats_per_bar`.

    >>> proc = DBNBarTrackingProcessor(beats_per_bar=[3, 4])
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.downbeats.DBNBarTrackingProcessor object at 0x...>

    Call this DBNDownBeatTrackingProcessor with beat positions and downbeat
    activation function returned by RNNBarProcessor to obtain the positions.

    >>> beats = np.loadtxt('tests/data/detections/sample.dbn_beat_tracker.txt')
    >>> act = RNNBarProcessor()(('tests/data/audio/sample.wav', beats))
    >>> proc(act)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[0.1 , 1. ],
           [0.45, 2. ],
           [0.8 , 3. ],
           [1.12, 1. ],
           [1.48, 2. ],
           [1.8 , 3. ],
           [2.15, 1. ],
           [2.49, 2. ]])

    """
    OBSERVATION_WEIGHT = 100
    METER_CHANGE_PROB = 1e-07

    def __init__(self, beats_per_bar=(3, 4), observation_weight=OBSERVATION_WEIGHT, meter_change_prob=METER_CHANGE_PROB, **kwargs):
        self.beats_per_bar = beats_per_bar
        state_spaces = []
        transition_models = []
        for beats in self.beats_per_bar:
            st = BarStateSpace(beats, min_interval=1, max_interval=1)
            tm = BarTransitionModel(st, transition_lambda=1)
            state_spaces.append(st)
            transition_models.append(tm)
        self.st = MultiPatternStateSpace(state_spaces)
        self.tm = MultiPatternTransitionModel(transition_models, transition_prob=meter_change_prob)
        self.om = RNNBeatTrackingObservationModel(self.st, observation_weight)
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)

    def process(self, data, **kwargs):
        """
        Detect downbeats from the given beats and activation function with
        Viterbi decoding.

        Parameters
        ----------
        data : numpy array, shape (num_beats, 2)
            Array containing beat positions (first column) and corresponding
            downbeat activations (second column).

        Returns
        -------
        numpy array, shape (num_beats, 2)
            Decoded (down-)beat positions and beat numbers.

        Notes
        -----
        The position of the last beat is not decoded, but rather extrapolated
        based on the position and meter of the second to last beat.

        """
        beats = data[:, 0]
        activations = data[:, 1]
        activations = activations[:-1]
        path, _ = self.hmm.viterbi(activations)
        position = self.st.state_positions[path]
        beat_numbers = position.astype(int) + 1
        meter = self.beats_per_bar[self.st.state_patterns[path[-1]]]
        last_beat_number = np.mod(beat_numbers[-1], meter) + 1
        beat_numbers = np.append(beat_numbers, last_beat_number)
        return np.vstack(zip(beats, beat_numbers))

    @classmethod
    def add_arguments(cls, parser, beats_per_bar, observation_weight=OBSERVATION_WEIGHT, meter_change_prob=METER_CHANGE_PROB):
        """
        Add DBN related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        beats_per_bar : int or list, optional
            Number of beats per bar to be modeled. Can be either a single
            number or a list with bar lengths (in beats).
        observation_weight : float, optional
            Weight for the activations at downbeat times.
        meter_change_prob : float, optional
            Probability to change meter at bar boundaries.

        Returns
        -------
        parser_group : argparse argument group
            DBN bar tracking argument parser group

        """
        from ..utils import OverrideDefaultListAction
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        g.add_argument('--beats_per_bar', action=OverrideDefaultListAction, default=beats_per_bar, type=int, sep=',', help='number of beats per bar to be modeled (comma separated list of bar length in beats) [default=%(default)s]')
        g.add_argument('--observation_weight', action='store', type=float, default=observation_weight, help='weight for the downbeat activations [default=%(default)i]')
        g.add_argument('--meter_change_prob', action='store', type=float, default=meter_change_prob, help='meter change probability [default=%(default).g]')
        parser = parser.add_argument_group('output arguments')
        parser.add_argument('--downbeats', action='store_true', default=False, help='output only the downbeats')
        return parser