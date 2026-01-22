import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def _all_pseudo(self, first_alphabet, second_alphabet):
    """Return a dictionary with all counts set to a default value (PRIVATE).

        This takes the letters in first alphabet and second alphabet and
        creates a dictionary with keys of two tuples organized as:
        (letter of first alphabet, letter of second alphabet). The values
        are all set to the value of the class attribute DEFAULT_PSEUDO.
        """
    all_counts = {}
    for first_state in first_alphabet:
        for second_state in second_alphabet:
            all_counts[first_state, second_state] = self.DEFAULT_PSEUDO
    return all_counts