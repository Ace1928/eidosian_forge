import math
import warnings
from .DynamicProgramming import ScaledDPAlgorithms
from Bio import BiopythonDeprecationWarning
class TrainingSequence:
    """Hold a training sequence with emissions and optionally, a state path."""

    def __init__(self, emissions, state_path):
        """Initialize a training sequence.

        Arguments:
         - emissions - An iterable (e.g., a tuple, list, or Seq object)
           containing the sequence of emissions in the training sequence.
         - state_path - An iterable (e.g., a tuple or list) containing the
           sequence of states. If there is no known state path, then the
           sequence of states should be an empty iterable.

        """
        if len(state_path) > 0 and len(emissions) != len(state_path):
            raise ValueError('State path does not match associated emissions.')
        self.emissions = emissions
        self.states = state_path