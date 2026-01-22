import warnings
from Bio import BiopythonDeprecationWarning
class ScaledDPAlgorithms(AbstractDPAlgorithms):
    """Implement forward and backward algorithms using a rescaling approach.

    This scales the f and b variables, so that they remain within a
    manageable numerical interval during calculations. This approach is
    described in Durbin et al. on p 78.

    This approach is a little more straightforward then log transformation
    but may still give underflow errors for some types of models. In these
    cases, the LogDPAlgorithms class should be used.
    """

    def __init__(self, markov_model, sequence):
        """Initialize the scaled approach to calculating probabilities.

        Arguments:
         - markov_model -- The current Markov model we are working with.
         - sequence -- A TrainingSequence object that must have a
           set of emissions to work with.

        """
        AbstractDPAlgorithms.__init__(self, markov_model, sequence)
        self._s_values = {}

    def _calculate_s_value(self, seq_pos, previous_vars):
        """Calculate the next scaling variable for a sequence position (PRIVATE).

        This utilizes the approach of choosing s values such that the
        sum of all of the scaled f values is equal to 1.

        Arguments:
         - seq_pos -- The current position we are at in the sequence.
         - previous_vars -- All of the forward or backward variables
           calculated so far.

        Returns:
         - The calculated scaling variable for the sequence item.

        """
        state_letters = self._mm.state_alphabet
        s_value = 0
        for main_state in state_letters:
            emission = self._mm.emission_prob[main_state, self._seq.emissions[seq_pos]]
            trans_and_var_sum = 0
            for second_state in self._mm.transitions_from(main_state):
                var_value = previous_vars[second_state, seq_pos - 1]
                trans_value = self._mm.transition_prob[second_state, main_state]
                trans_and_var_sum += var_value * trans_value
            s_value += emission * trans_and_var_sum
        return s_value

    def _forward_recursion(self, cur_state, sequence_pos, forward_vars):
        """Calculate the value of the forward recursion (PRIVATE).

        Arguments:
         - cur_state -- The letter of the state we are calculating the
           forward variable for.
         - sequence_pos -- The position we are at in the training seq.
         - forward_vars -- The current set of forward variables

        """
        if sequence_pos not in self._s_values:
            self._s_values[sequence_pos] = self._calculate_s_value(sequence_pos, forward_vars)
        seq_letter = self._seq.emissions[sequence_pos]
        cur_emission_prob = self._mm.emission_prob[cur_state, seq_letter]
        scale_emission_prob = cur_emission_prob / self._s_values[sequence_pos]
        state_pos_sum = 0
        have_transition = 0
        for second_state in self._mm.transitions_from(cur_state):
            have_transition = 1
            prev_forward = forward_vars[second_state, sequence_pos - 1]
            cur_trans_prob = self._mm.transition_prob[second_state, cur_state]
            state_pos_sum += prev_forward * cur_trans_prob
        if have_transition:
            return scale_emission_prob * state_pos_sum
        else:
            return None

    def _backward_recursion(self, cur_state, sequence_pos, backward_vars):
        """Calculate the value of the backward recursion (PRIVATE).

        Arguments:
         - cur_state -- The letter of the state we are calculating the
           forward variable for.
         - sequence_pos -- The position we are at in the training seq.
         - backward_vars -- The current set of backward variables

        """
        if sequence_pos not in self._s_values:
            self._s_values[sequence_pos] = self._calculate_s_value(sequence_pos, backward_vars)
        state_pos_sum = 0
        have_transition = 0
        for second_state in self._mm.transitions_from(cur_state):
            have_transition = 1
            seq_letter = self._seq.emissions[sequence_pos + 1]
            cur_emission_prob = self._mm.emission_prob[cur_state, seq_letter]
            prev_backward = backward_vars[second_state, sequence_pos + 1]
            cur_transition_prob = self._mm.transition_prob[cur_state, second_state]
            state_pos_sum += cur_emission_prob * prev_backward * cur_transition_prob
        if have_transition:
            return state_pos_sum / self._s_values[sequence_pos]
        else:
            return None