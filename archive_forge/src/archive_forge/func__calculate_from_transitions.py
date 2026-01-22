import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def _calculate_from_transitions(trans_probs):
    """Calculate which 'from transitions' are allowed for each state (PRIVATE).

    This looks through all of the trans_probs, and uses this dictionary
    to determine allowed transitions. It converts this information into
    a dictionary, whose keys are source states and whose values are
    lists of destination states reachable from the source state via a
    transition.
    """
    transitions = defaultdict(list)
    for from_state, to_state in trans_probs:
        transitions[from_state].append(to_state)
    return transitions