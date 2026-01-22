from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.ml.hmm import ObservationModel, TransitionModel
class MultiPatternTransitionModel(TransitionModel):
    """
    Transition model for pattern tracking with a HMM.

    Add transitions with the given probability between the individual
    transition models. These transition models must correspond to the state
    spaces forming a :class:`MultiPatternStateSpace`.

    Parameters
    ----------
    transition_models : list
        List with :class:`TransitionModel` instances.
    transition_prob : numpy array or float, optional
        Probabilities to change the pattern at pattern boundaries. If an array
        is given, the first dimension corresponds to the origin pattern, the
        second to the destination pattern. If a single value is given, a
        uniform transition distribution to all other patterns is assumed. Set
        to None to stay within the same pattern.

    """

    def __init__(self, transition_models, transition_prob=None):
        self.transition_models = transition_models
        self.transition_prob = transition_prob
        num_patterns = len(transition_models)
        first_states = []
        last_states = []
        for p, tm in enumerate(self.transition_models):
            offset = 0
            if p == 0:
                states = tm.states
                pointers = tm.pointers
                probabilities = tm.probabilities
            else:
                offset = len(pointers) - 1
                states = np.hstack((states, tm.states + len(pointers) - 1))
                pointers = np.hstack((pointers, tm.pointers[1:] + max(pointers)))
                probabilities = np.hstack((probabilities, tm.probabilities))
            first_states.append(tm.state_space.first_states[0] + offset)
            last_states.append(tm.state_space.last_states[-1] + offset)
        states, prev_states, probabilities = self.make_dense(states, pointers, probabilities)
        if isinstance(transition_prob, float) and transition_prob:
            self.transition_prob = np.ones((num_patterns, num_patterns))
            self.transition_prob *= transition_prob / (num_patterns - 1)
            diag = np.diag_indices_from(self.transition_prob)
            self.transition_prob[diag] = 1.0 - transition_prob
        else:
            self.transition_prob = transition_prob
        if self.transition_prob is not None and num_patterns > 1:
            new_states = []
            new_prev_states = []
            new_probabilities = []
            for p in range(num_patterns):
                idx = np.logical_and(np.in1d(prev_states, last_states[p]), np.in1d(states, first_states[p]))
                prob = probabilities[idx]
                probabilities[idx] *= self.transition_prob[p, p]
                for p_ in np.setdiff1d(range(num_patterns), p):
                    idx_ = np.logical_and(np.in1d(prev_states, last_states[p_]), np.in1d(states, first_states[p_]))
                    if len(np.nonzero(idx)[0]) != len(np.nonzero(idx_)[0]):
                        raise ValueError('Cannot add transition between patterns with different number of entering/exiting states.')
                    new_states.extend(states[idx])
                    new_prev_states.extend(prev_states[idx_])
                    new_probabilities.extend(prob * self.transition_prob[p, p_])
            states = np.append(states, new_states)
            prev_states = np.append(prev_states, new_prev_states)
            probabilities = np.append(probabilities, new_probabilities)
        transitions = self.make_sparse(states, prev_states, probabilities)
        super(MultiPatternTransitionModel, self).__init__(*transitions)