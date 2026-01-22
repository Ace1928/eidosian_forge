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
def _add_multi_graph_node(self, node):
    """
        Args:
            node (DAGDepNode): considered node.

        Returns:
            node_id(int): corresponding label to the added node.
        """
    node_id = self._multi_graph.add_node(node)
    node.node_id = node_id
    return node_id