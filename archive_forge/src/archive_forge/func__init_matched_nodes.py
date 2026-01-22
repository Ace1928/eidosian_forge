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
def _init_matched_nodes(self):
    """
        Initialize the list of current matched nodes.
        """
    self.matched_nodes_list.append([self.node_id_c, self.circuit_dag.get_node(self.node_id_c), self.successors_to_visit[self.node_id_c]])