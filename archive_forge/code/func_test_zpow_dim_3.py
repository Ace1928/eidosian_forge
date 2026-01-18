import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_zpow_dim_3():
    L = np.exp(2 * np.pi * 1j / 3)
    L2 = L ** 2
    z = cirq.ZPowGate(dimension=3)
    assert cirq.Z != z
    expected = [[1, 0, 0], [0, L, 0], [0, 0, L2]]
    assert np.allclose(cirq.unitary(z), expected)
    sim = cirq.Simulator()
    circuit = cirq.Circuit([z(cirq.LineQid(0, 3)) ** 0.5] * 6)
    svs = [step.state_vector(copy=True) for step in sim.simulate_moment_steps(circuit, initial_state=0)]
    expected = [[1, 0, 0]] * 6
    assert np.allclose(svs, expected)
    svs = [step.state_vector(copy=True) for step in sim.simulate_moment_steps(circuit, initial_state=1)]
    expected = [[0, L ** 0.5, 0], [0, L ** 1.0, 0], [0, L ** 1.5, 0], [0, L ** 2.0, 0], [0, L ** 2.5, 0], [0, 1, 0]]
    assert np.allclose(svs, expected)
    svs = [step.state_vector(copy=True) for step in sim.simulate_moment_steps(circuit, initial_state=2)]
    expected = [[0, 0, L], [0, 0, L2], [0, 0, 1], [0, 0, L], [0, 0, L2], [0, 0, 1]]
    assert np.allclose(svs, expected)