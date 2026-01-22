from __future__ import annotations
from collections.abc import Iterable
import logging
from qiskit.circuit import Qubit, Clbit, Instruction
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
def _apply_scheduled_op(self, dag: DAGCircuit, t_start: int, oper: Instruction, qubits: Qubit | Iterable[Qubit], clbits: Clbit | Iterable[Clbit]=()):
    """Add new operation to DAG with scheduled information.

        This is identical to apply_operation_back + updating the node_start_time propety.

        Args:
            dag: DAG circuit on which the sequence is applied.
            t_start: Start time of new node.
            oper: New operation that is added to the DAG circuit.
            qubits: The list of qubits that the operation acts on.
            clbits: The list of clbits that the operation acts on.
        """
    if isinstance(qubits, Qubit):
        qubits = [qubits]
    if isinstance(clbits, Clbit):
        clbits = [clbits]
    new_node = dag.apply_operation_back(oper, qargs=qubits, cargs=clbits, check=False)
    self.property_set['node_start_time'][new_node] = t_start