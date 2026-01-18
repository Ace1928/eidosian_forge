import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_simulate_initial_state():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.CliffordSimulator()
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit()
            if b0:
                circuit.append(cirq.X(q0))
            if b1:
                circuit.append(cirq.X(q1))
            circuit.append(cirq.measure(q0, q1))
            result = simulator.simulate(circuit, initial_state=1)
            expected_state = np.zeros(shape=(2, 2))
            expected_state[b0][1 - b1] = 1.0
            np.testing.assert_almost_equal(result.final_state.to_numpy(), np.reshape(expected_state, 4))