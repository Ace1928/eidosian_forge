import math
import warnings
from .DynamicProgramming import ScaledDPAlgorithms
from Bio import BiopythonDeprecationWarning
def update_emissions(self, emission_counts, training_seq, forward_vars, backward_vars, training_seq_prob):
    """Add the contribution of a new training sequence to the emissions.

        Arguments:
         - emission_counts -- A dictionary of the current counts for the
           emissions
         - training_seq -- The training sequence we are working with
         - forward_vars -- Probabilities calculated using the forward
           algorithm.
         - backward_vars -- Probabilities calculated using the backwards
           algorithm.
         - training_seq_prob - The probability of the current sequence.

        This calculates E_{k}(b) (the estimated emission probability for
        emission letter b from state k) using formula 3.21 in Durbin et al.

        """
    for k in self._markov_model.state_alphabet:
        for b in self._markov_model.emission_alphabet:
            expected_times = 0
            for i in range(len(training_seq.emissions)):
                if training_seq.emissions[i] == b:
                    expected_times += forward_vars[k, i] * backward_vars[k, i]
            emission_counts[k, b] += expected_times / training_seq_prob
    return emission_counts