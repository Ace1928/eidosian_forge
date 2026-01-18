import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def transitions_from(self, state_letter):
    """Get all destination states which can transition from source state_letter.

        This returns all letters which the given state_letter can transition
        to, i.e. all the destination states reachable from state_letter.

        An empty list is returned if state_letter has no outgoing transitions.
        """
    if state_letter in self._transitions_from:
        return self._transitions_from[state_letter]
    else:
        return []