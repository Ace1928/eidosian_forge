from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.ml.hmm import ObservationModel, TransitionModel
class MultiPatternStateSpace(object):
    """
    State space for rhythmic pattern tracking with a HMM.

    Model a joint state space with the given `state_spaces` by stacking the
    individual state spaces.

    Parameters
    ----------
    state_spaces : list
        List with state spaces to model.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, state_spaces):
        self.num_patterns = len(state_spaces)
        self.state_spaces = state_spaces
        self.state_positions = np.empty(0)
        self.state_intervals = np.empty(0, dtype=np.int)
        self.state_patterns = np.empty(0, dtype=np.int)
        self.num_states = 0
        self.first_states = []
        self.last_states = []
        for p, pss in enumerate(state_spaces):
            self.state_positions = np.hstack((self.state_positions, pss.state_positions))
            self.state_intervals = np.hstack((self.state_intervals, pss.state_intervals))
            self.state_patterns = np.hstack((self.state_patterns, np.repeat(p, pss.num_states)))
            self.first_states.append(pss.first_states[0] + self.num_states)
            self.last_states.append(pss.last_states[-1] + self.num_states)
            self.num_states += pss.num_states