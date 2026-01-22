import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
class MaximalMatches:
    """
    Class MaximalMatches allows to sort and store the maximal matches from the list
    of matches obtained with the pattern matching algorithm.
    """

    def __init__(self, pattern_matches):
        """Initialize MaximalMatches with the necessary arguments.
        Args:
            pattern_matches (list): list of matches obtained from running the algorithm.
        """
        self.pattern_matches = pattern_matches
        self.max_match_list = []

    def run_maximal_matches(self):
        """Method that extracts and stores maximal matches in decreasing length order."""
        self.max_match_list = [Match(sorted(self.pattern_matches[0].match), self.pattern_matches[0].qubit)]
        for matches in self.pattern_matches[1:]:
            present = False
            for max_match in self.max_match_list:
                for elem in matches.match:
                    if elem in max_match.match and len(matches.match) <= len(max_match.match):
                        present = True
            if not present:
                self.max_match_list.append(Match(sorted(matches.match), matches.qubit))