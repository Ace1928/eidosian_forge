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
def _find_forward_candidates(self, node_id_p):
    """Find the candidate nodes to be matched in the pattern for a given node in the pattern.
        Args:
            node_id_p (int): Node ID in pattern.
        """
    matches = [i[0] for i in self.match]
    pred = matches.copy()
    if len(pred) > 1:
        pred.sort()
    pred.remove(node_id_p)
    if self.pattern_dag.direct_successors(node_id_p):
        maximal_index = self.pattern_dag.direct_successors(node_id_p)[-1]
        for elem in pred:
            if elem > maximal_index:
                pred.remove(elem)
    block = []
    for node_id in pred:
        for dir_succ in self.pattern_dag.direct_successors(node_id):
            if dir_succ not in matches:
                succ = self.pattern_dag.successors(dir_succ)
                block = block + succ
    self.candidates = list(set(self.pattern_dag.direct_successors(node_id_p)) - set(matches) - set(block))