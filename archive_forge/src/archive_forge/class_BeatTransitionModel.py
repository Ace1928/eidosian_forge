from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.ml.hmm import ObservationModel, TransitionModel
class BeatTransitionModel(TransitionModel):
    """
    Transition model for beat tracking with a HMM.

    Within the beat the tempo stays the same; at beat boundaries transitions
    from one tempo (i.e. interval) to another are allowed, following an
    exponential distribution.

    Parameters
    ----------
    state_space : :class:`BeatStateSpace` instance
        BeatStateSpace instance.
    transition_lambda : float
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, state_space, transition_lambda):
        self.state_space = state_space
        self.transition_lambda = float(transition_lambda)
        states = np.arange(state_space.num_states, dtype=np.uint32)
        states = np.setdiff1d(states, state_space.first_states)
        prev_states = states - 1
        probabilities = np.ones_like(states, dtype=np.float)
        to_states = state_space.first_states
        from_states = state_space.last_states
        from_int = state_space.state_intervals[from_states]
        to_int = state_space.state_intervals[to_states]
        prob = exponential_transition(from_int, to_int, self.transition_lambda)
        from_prob, to_prob = np.nonzero(prob)
        states = np.hstack((states, to_states[to_prob]))
        prev_states = np.hstack((prev_states, from_states[from_prob]))
        probabilities = np.hstack((probabilities, prob[prob != 0]))
        transitions = self.make_sparse(states, prev_states, probabilities)
        super(BeatTransitionModel, self).__init__(*transitions)