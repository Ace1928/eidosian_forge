import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('gate, num_copies', [(cirq.X, 1), (cirq.Y, 2), (cirq.Z, 3), (cirq.H, 4)])
def test_parallel_gate_operation_is_consistent(gate, num_copies):
    cirq.testing.assert_implements_consistent_protocols(cirq.ParallelGate(gate, num_copies))