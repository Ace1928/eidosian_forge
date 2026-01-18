import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('split', [True, False])
def test_simulate_moment_steps_intermediate_measurement(split):
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0), cirq.H(q0))
    simulator = cirq.CliffordSimulator(split_untangled_states=split)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 1:
            result = int(step.measurements['q(0)'][0])
            expected = np.zeros(2)
            expected[result] = 1
            np.testing.assert_almost_equal(step.state.to_numpy(), expected)
        if i == 2:
            expected = np.array([np.sqrt(0.5), np.sqrt(0.5) * (-1) ** result])
            np.testing.assert_almost_equal(step.state.to_numpy(), expected)