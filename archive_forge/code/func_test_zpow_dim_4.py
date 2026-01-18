import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_zpow_dim_4():
    z = cirq.ZPowGate(dimension=4)
    assert cirq.Z != z
    expected = [[1, 0, 0, 0], [0, 1j, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1j]]
    assert np.allclose(cirq.unitary(z), expected)
    sim = cirq.Simulator()
    circuit = cirq.Circuit([z(cirq.LineQid(0, 4)) ** 0.5] * 8)
    svs = [step.state_vector(copy=True) for step in sim.simulate_moment_steps(circuit, initial_state=0)]
    expected = [[1, 0, 0, 0]] * 8
    assert np.allclose(svs, expected)
    svs = [step.state_vector(copy=True) for step in sim.simulate_moment_steps(circuit, initial_state=1)]
    expected = [[0, 1j ** 0.5, 0, 0], [0, 1j ** 1.0, 0, 0], [0, 1j ** 1.5, 0, 0], [0, 1j ** 2.0, 0, 0], [0, 1j ** 2.5, 0, 0], [0, 1j ** 3.0, 0, 0], [0, 1j ** 3.5, 0, 0], [0, 1, 0, 0]]
    assert np.allclose(svs, expected)
    svs = [step.state_vector(copy=True) for step in sim.simulate_moment_steps(circuit, initial_state=2)]
    expected = [[0, 0, 1j, 0], [0, 0, -1, 0], [0, 0, -1j, 0], [0, 0, 1, 0], [0, 0, 1j, 0], [0, 0, -1, 0], [0, 0, -1j, 0], [0, 0, 1, 0]]
    assert np.allclose(svs, expected)
    svs = [step.state_vector(copy=True) for step in sim.simulate_moment_steps(circuit, initial_state=3)]
    expected = [[0, 0, 0, 1j ** 1.5], [0, 0, 0, 1j ** 3], [0, 0, 0, 1j ** 0.5], [0, 0, 0, 1j ** 2], [0, 0, 0, 1j ** 3.5], [0, 0, 0, 1j ** 1], [0, 0, 0, 1j ** 2.5], [0, 0, 0, 1]]
    assert np.allclose(svs, expected)