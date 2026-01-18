from typing import Sequence
import numpy as np
import pytest
import cirq
def test_state_tomography_diagonal():
    n = 2
    qubits = cirq.LineQubit.range(n)
    for state in range(2 ** n):
        circuit = cirq.Circuit()
        for i, q in enumerate(qubits):
            bit = state & 1 << n - i - 1
            if bit:
                circuit.append(cirq.X(q))
        res = cirq.experiments.state_tomography(cirq.Simulator(seed=87539319), qubits, circuit, repetitions=1000, prerotations=[(0, 0), (0, 0.5), (0.5, 0.5)])
        should_be = np.zeros((2 ** n, 2 ** n))
        should_be[state, state] = 1
        assert np.allclose(res.data, should_be, atol=0.05)