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
def _get_node_forward(self, list_id):
    """
        Return node and successors from the matched_nodes_list for a given ID.
        Args:
            list_id (int): considered list id of the desired node.
        Returns:
            CommutationDAGNode: Node from the matched_node_list.
            list(int): List of successors.
        """
    node = self.matched_nodes_list[list_id][1]
    succ = self.matched_nodes_list[list_id][2]
    return (node, succ)