import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def _calculate_emissions(emission_probs):
    """Calculate which symbols can be emitted in each state (PRIVATE)."""
    emissions = defaultdict(list)
    for state, symbol in emission_probs:
        emissions[state].append(symbol)
    return emissions