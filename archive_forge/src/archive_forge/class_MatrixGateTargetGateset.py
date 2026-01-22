import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers.optimize_for_target_gateset import _decompose_operations_to_target_gateset
import pytest
class MatrixGateTargetGateset(cirq.CompilationTargetGateset):

    def __init__(self):
        super().__init__(cirq.MatrixGate)

    @property
    def num_qubits(self) -> int:
        return 2

    def decompose_to_target_gateset(self, op: 'cirq.Operation', _) -> DecomposeResult:
        if cirq.num_qubits(op) != 2 or not cirq.has_unitary(op):
            return NotImplemented
        return cirq.MatrixGate(cirq.unitary(op), name='M').on(*op.qubits)