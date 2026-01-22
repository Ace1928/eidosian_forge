from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.ml.hmm import ObservationModel, TransitionModel
class BeatStateSpace(object):
    """
    State space for beat tracking with a HMM.

    Parameters
    ----------
    min_interval : float
        Minimum interval to model.
    max_interval : float
        Maximum interval to model.
    num_intervals : int, optional
        Number of intervals to model; if set, limit the number of intervals
        and use a log spacing instead of the default linear spacing.

    Attributes
    ----------
    num_states : int
        Number of states.
    intervals : numpy array
        Modeled intervals.
    num_intervals : int
        Number of intervals.
    state_positions : numpy array
        Positions of the states (i.e. 0...1).
    state_intervals : numpy array
        Intervals of the states (i.e. 1 / tempo).
    first_states : numpy array
        First state of each interval.
    last_states : numpy array
        Last state of each interval.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, min_interval, max_interval, num_intervals=None):
        intervals = np.arange(np.round(min_interval), np.round(max_interval) + 1)
        if num_intervals is not None and num_intervals < len(intervals):
            num_log_intervals = num_intervals
            intervals = []
            while len(intervals) < num_intervals:
                intervals = np.logspace(np.log2(min_interval), np.log2(max_interval), num_log_intervals, base=2)
                intervals = np.unique(np.round(intervals))
                num_log_intervals += 1
        self.intervals = np.ascontiguousarray(intervals, dtype=np.int)
        self.num_states = int(np.sum(intervals))
        self.num_intervals = len(intervals)
        first_states = np.cumsum(np.r_[0, self.intervals[:-1]])
        self.first_states = first_states.astype(np.int)
        self.last_states = np.cumsum(self.intervals) - 1
        self.state_positions = np.empty(self.num_states)
        self.state_intervals = np.empty(self.num_states, dtype=np.int)
        idx = 0
        for i in self.intervals:
            self.state_positions[idx:idx + i] = np.linspace(0, 1, i, endpoint=False)
            self.state_intervals[idx:idx + i] = i
            idx += i