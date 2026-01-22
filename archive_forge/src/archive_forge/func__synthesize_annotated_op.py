from typing import Optional, Union, List, Tuple
import rustworkx as rx
from qiskit.circuit.operation import Operation
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ControlFlowOp, ControlledGate, EquivalenceLibrary
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper
from qiskit.circuit.annotated_operation import (
from qiskit.synthesis.clifford import (
from qiskit.synthesis.linear import synth_cnot_count_full_pmh, synth_cnot_depth_line_kms
from qiskit.synthesis.permutation import (
from .plugin import HighLevelSynthesisPluginManager, HighLevelSynthesisPlugin
def _synthesize_annotated_op(self, op: Operation) -> Union[Operation, None]:
    """
        Recursively synthesizes annotated operations.
        Returns either the synthesized operation or None (which occurs when the operation
        is not an annotated operation).
        """
    if isinstance(op, AnnotatedOperation):
        synthesized_op, _ = self._recursively_handle_op(op.base_op, qubits=None)
        if isinstance(synthesized_op, AnnotatedOperation):
            raise TranspilerError('HighLevelSynthesis failed to synthesize the base operation of an annotated operation.')
        for modifier in op.modifiers:
            if isinstance(synthesized_op, DAGCircuit):
                synthesized_op = dag_to_circuit(synthesized_op, copy_operations=False)
            if isinstance(modifier, InverseModifier):
                synthesized_op = synthesized_op.inverse()
            elif isinstance(modifier, ControlModifier):
                if isinstance(synthesized_op, QuantumCircuit):
                    synthesized_op = synthesized_op.to_gate()
                synthesized_op = synthesized_op.control(num_ctrl_qubits=modifier.num_ctrl_qubits, label=None, ctrl_state=modifier.ctrl_state, annotated=False)
                if isinstance(synthesized_op, AnnotatedOperation):
                    raise TranspilerError('HighLevelSynthesis failed to synthesize the control modifier.')
                synthesized_op, _ = self._recursively_handle_op(synthesized_op)
            elif isinstance(modifier, PowerModifier):
                if isinstance(synthesized_op, QuantumCircuit):
                    qc = synthesized_op
                else:
                    qc = QuantumCircuit(synthesized_op.num_qubits, synthesized_op.num_clbits)
                    qc.append(synthesized_op, range(synthesized_op.num_qubits), range(synthesized_op.num_clbits))
                qc = qc.power(modifier.power)
                synthesized_op = qc.to_gate()
                synthesized_op, _ = self._recursively_handle_op(synthesized_op)
            else:
                raise TranspilerError(f'Unknown modifier {modifier}.')
        return synthesized_op
    return None