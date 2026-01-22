import math
import warnings
from .DynamicProgramming import ScaledDPAlgorithms
from Bio import BiopythonDeprecationWarning
class AbstractTrainer:
    """Provide generic functionality needed in all trainers."""

    def __init__(self, markov_model):
        """Initialize the class."""
        self._markov_model = markov_model

    def log_likelihood(self, probabilities):
        """Calculate the log likelihood of the training seqs.

        Arguments:
         - probabilities -- A list of the probabilities of each training
           sequence under the current parameters, calculated using the
           forward algorithm.

        """
        total_likelihood = 0
        for probability in probabilities:
            total_likelihood += math.log(probability)
        return total_likelihood

    def estimate_params(self, transition_counts, emission_counts):
        """Get a maximum likelihood estimation of transition and emmission.

        Arguments:
         - transition_counts -- A dictionary with the total number of counts
           of transitions between two states.
         - emissions_counts -- A dictionary with the total number of counts
           of emmissions of a particular emission letter by a state letter.

        This then returns the maximum likelihood estimators for the
        transitions and emissions, estimated by formulas 3.18 in
        Durbin et al::

            a_{kl} = A_{kl} / sum(A_{kl'})
            e_{k}(b) = E_{k}(b) / sum(E_{k}(b'))

        Returns:
        Transition and emission dictionaries containing the maximum
        likelihood estimators.

        """
        ml_transitions = self.ml_estimator(transition_counts)
        ml_emissions = self.ml_estimator(emission_counts)
        return (ml_transitions, ml_emissions)

    def ml_estimator(self, counts):
        """Calculate the maximum likelihood estimator.

        This can calculate maximum likelihoods for both transitions
        and emissions.

        Arguments:
         - counts -- A dictionary of the counts for each item.

        See estimate_params for a description of the formula used for
        calculation.

        """
        all_ordered = sorted(counts)
        ml_estimation = {}
        cur_letter = None
        cur_letter_counts = 0
        for cur_item in all_ordered:
            if cur_item[0] != cur_letter:
                cur_letter = cur_item[0]
                cur_letter_counts = counts[cur_item]
                cur_position = all_ordered.index(cur_item) + 1
                while cur_position < len(all_ordered) and all_ordered[cur_position][0] == cur_item[0]:
                    cur_letter_counts += counts[all_ordered[cur_position]]
                    cur_position += 1
            else:
                pass
            cur_ml = counts[cur_item] / cur_letter_counts
            ml_estimation[cur_item] = cur_ml
        return ml_estimation