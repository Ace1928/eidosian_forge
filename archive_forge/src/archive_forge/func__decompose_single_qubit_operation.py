from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def _decompose_single_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
    return cirq.X(*op.qubits) if op.gate == cirq.Y else NotImplemented