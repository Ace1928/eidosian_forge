import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_simulate_moment_steps_sample():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    simulator = ccq.mps_simulator.MPSSimulator()
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            np.testing.assert_almost_equal(step._simulator_state().to_numpy(), np.asarray([1.0 / math.sqrt(2), 0.0, 1.0 / math.sqrt(2), 0.0]))
            assert len(str(step).split('Tensor(')) == 3
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert np.array_equal(sample, [True, False]) or np.array_equal(sample, [False, False])
            np.testing.assert_almost_equal(step._simulator_state().to_numpy(), np.asarray([1.0 / math.sqrt(2), 0.0, 1.0 / math.sqrt(2), 0.0]))
        else:
            np.testing.assert_almost_equal(step._simulator_state().to_numpy(), np.asarray([1.0 / math.sqrt(2), 0.0, 0.0, 1.0 / math.sqrt(2)]))
            assert len(str(step).split('Tensor(')) == 3
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert np.array_equal(sample, [True, True]) or np.array_equal(sample, [False, False])