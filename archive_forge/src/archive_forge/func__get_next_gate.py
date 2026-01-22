from __future__ import annotations
from collections.abc import Generator
from qiskit.circuit.gate import Gate
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGOutNode
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
@classmethod
def _get_next_gate(cls, dag: DAGCircuit, node: DAGOpNode) -> Generator[DAGOpNode, None, None]:
    """Get next non-delay nodes.

        Args:
            dag: DAG circuit to be rescheduled with constraints.
            node: Current node.

        Returns:
            A list of non-delay successors.
        """
    for next_node in dag.successors(node):
        if not isinstance(next_node, DAGOutNode):
            yield next_node