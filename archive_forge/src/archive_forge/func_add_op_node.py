import math
import heapq
from collections import OrderedDict, defaultdict
import rustworkx as rx
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.dagcircuit.exceptions import DAGDependencyError
from qiskit.dagcircuit.dagdepnode import DAGDepNode
def add_op_node(self, operation, qargs, cargs):
    """Add a DAGDepNode to the graph and update the edges.

        Args:
            operation (qiskit.circuit.Operation): operation as a quantum gate
            qargs (list[~qiskit.circuit.Qubit]): list of qubits on which the operation acts
            cargs (list[Clbit]): list of classical wires to attach to
        """
    new_node = self._create_op_node(operation, qargs, cargs)
    self._add_multi_graph_node(new_node)
    self._update_edges()