from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.ml.hmm import ObservationModel, TransitionModel
class BarStateSpace(object):
    """
    State space for bar tracking with a HMM.

    Model `num_beat` identical beats with the given arguments in a single state
    space.

    Parameters
    ----------
    num_beats : int
        Number of beats to form a bar.
    min_interval : float
        Minimum beat interval to model.
    max_interval : float
        Maximum beat interval to model.
    num_intervals : int, optional
        Number of beat intervals to model; if set, limit the number of
        intervals and use a log spacing instead of the default linear spacing.

    Attributes
    ----------
    num_beats : int
        Number of beats.
    num_states : int
        Number of states.
    num_intervals : int
        Number of intervals.
    state_positions : numpy array
        Positions of the states.
    state_intervals : numpy array
        Intervals of the states.
    first_states : list
        First states of each beat.
    last_states : list
        Last states of each beat.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, num_beats, min_interval, max_interval, num_intervals=None):
        self.num_beats = int(num_beats)
        self.state_positions = np.empty(0)
        self.state_intervals = np.empty(0, dtype=np.int)
        self.num_states = 0
        self.first_states = []
        self.last_states = []
        bss = BeatStateSpace(min_interval, max_interval, num_intervals)
        for b in range(self.num_beats):
            self.state_positions = np.hstack((self.state_positions, bss.state_positions + b))
            self.state_intervals = np.hstack((self.state_intervals, bss.state_intervals))
            self.first_states.append(bss.first_states + self.num_states)
            self.last_states.append(bss.last_states + self.num_states)
            self.num_states += bss.num_states