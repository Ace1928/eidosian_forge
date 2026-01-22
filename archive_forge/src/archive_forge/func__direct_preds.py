from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
def _direct_preds(self, node):
    """Returns direct predecessors of a node. This function takes into account the
        direction of collecting blocks, that is node's predecessors when collecting
        backwards are the direct successors of a node in the DAG.
        """
    if not self.is_dag_dependency:
        if self._collect_from_back:
            return [pred for pred in self.dag.successors(node) if isinstance(pred, DAGOpNode)]
        else:
            return [pred for pred in self.dag.predecessors(node) if isinstance(pred, DAGOpNode)]
    elif self._collect_from_back:
        return [self.dag.get_node(pred_id) for pred_id in self.dag.direct_successors(node.node_id)]
    else:
        return [self.dag.get_node(pred_id) for pred_id in self.dag.direct_predecessors(node.node_id)]