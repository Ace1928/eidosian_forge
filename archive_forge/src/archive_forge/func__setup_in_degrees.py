from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
def _setup_in_degrees(self):
    """For an efficient implementation, for every node we keep the number of its
        unprocessed immediate predecessors (called ``_in_degree``). This ``_in_degree``
        is set up at the start and updated throughout the algorithm.
        A node is leaf (or input) node iff its ``_in_degree`` is 0.
        When a node is (marked as) collected, the ``_in_degree`` of each of its immediate
        successor is updated by subtracting 1.
        Additionally, ``_pending_nodes`` explicitly keeps the list of nodes whose
        ``_in_degree`` is 0.
        """
    self._pending_nodes = []
    self._in_degree = {}
    for node in self._op_nodes():
        deg = len(self._direct_preds(node))
        self._in_degree[node] = deg
        if deg == 0:
            self._pending_nodes.append(node)