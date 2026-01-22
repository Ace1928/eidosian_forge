import math
import warnings
from .DynamicProgramming import ScaledDPAlgorithms
from Bio import BiopythonDeprecationWarning
def _count_emissions(self, training_seq, emission_counts):
    """Add emissions from the training sequence to the current counts (PRIVATE).

        Arguments:
         - training_seq -- A TrainingSequence with states and emissions
           to get the counts from
         - emission_counts -- The current emission counts to add to.

        """
    for index in range(len(training_seq.emissions)):
        cur_state = training_seq.states[index]
        cur_emission = training_seq.emissions[index]
        try:
            emission_counts[cur_state, cur_emission] += 1
        except KeyError:
            raise KeyError(f'Unexpected emission ({cur_state}, {cur_emission})')
    return emission_counts