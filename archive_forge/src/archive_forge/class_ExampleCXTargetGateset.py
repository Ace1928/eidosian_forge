from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
class ExampleCXTargetGateset(cirq.TwoQubitCompilationTargetGateset):

    def __init__(self):
        super().__init__(cirq.AnyUnitaryGateFamily(1), cirq.CNOT)

    def _decompose_two_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
        if not cirq.has_unitary(op):
            return NotImplemented
        assert self._intermediate_result_tag in op.tags
        q0, q1 = op.qubits
        return [cirq.X.on_each(q0, q1), cirq.CNOT(q0, q1), cirq.Y.on_each(q0, q1), cirq.CNOT(q0, q1), cirq.Z.on_each(q0, q1)]

    def _decompose_single_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
        if not cirq.has_unitary(op):
            return NotImplemented
        assert self._intermediate_result_tag in op.tags
        op_untagged = op.untagged
        assert isinstance(op_untagged, cirq.CircuitOperation)
        return cirq.decompose(op_untagged.circuit) if len(op_untagged.circuit) == 1 else super()._decompose_single_qubit_operation(op, _)