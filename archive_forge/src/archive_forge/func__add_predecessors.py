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
def _add_predecessors(self):
    """
        Create the list of predecessors for each node. Update DAGDependency
        'predecessors' attribute. It has to be used when the DAGDependency() object
        is complete (i.e. converters).
        """
    for node_id in range(0, len(self._multi_graph)):
        self._multi_graph.get_node_data(node_id).predecessors = list(rx.ancestors(self._multi_graph, node_id))