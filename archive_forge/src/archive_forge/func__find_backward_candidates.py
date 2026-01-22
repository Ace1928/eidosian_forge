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
def _find_backward_candidates(self, pattern_blocked, matches):
    """Function which returns the list possible backward candidates in the pattern dag.
        Args:
            pattern_blocked (list(bool)): list of blocked nodes in the pattern circuit.
            matches (list(int)): list of matches.
        Returns:
            list(int): list of backward candidates ID.
        """
    pattern_block = []
    for node_id in range(self.node_id_p, self.pattern_dag.size):
        if pattern_blocked[node_id]:
            pattern_block.append(node_id)
    matches_pattern = sorted((match[0] for match in matches))
    successors = self.pattern_dag.get_node(self.node_id_p).successors
    potential = []
    for index in range(self.node_id_p + 1, self.pattern_dag.size):
        if index not in successors and index not in pattern_block:
            potential.append(index)
    candidates_indices = list(set(potential) - set(matches_pattern))
    candidates_indices = sorted(candidates_indices)
    candidates_indices.reverse()
    return candidates_indices