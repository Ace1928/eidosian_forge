import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_state_copy():
    sim = ccq.mps_simulator.MPSSimulator()
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.H(q))
    state_Ms = []
    for step in sim.simulate_moment_steps(circuit):
        state_Ms.append(step.state.M)
    for x, y in itertools.combinations(state_Ms, 2):
        assert len(x) == len(y)
        for i in range(len(x)):
            assert not np.shares_memory(x[i], y[i])