import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_simulate_no_circuit():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit()
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_state.to_numpy(), np.array([1, 0, 0, 0]))
    assert len(result.measurements) == 0