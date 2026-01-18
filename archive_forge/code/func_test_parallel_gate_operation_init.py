import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('gate, num_copies, qubits', [(cirq.testing.SingleQubitGate(), 2, cirq.LineQubit.range(2)), (cirq.X ** 0.5, 4, cirq.LineQubit.range(4))])
def test_parallel_gate_operation_init(gate, num_copies, qubits):
    v = cirq.ParallelGate(gate, num_copies)
    assert v.sub_gate == gate
    assert v.num_copies == num_copies
    assert v.on(*qubits).qubits == tuple(qubits)