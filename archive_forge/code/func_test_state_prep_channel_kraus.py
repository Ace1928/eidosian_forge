import numpy as np
import cirq
import pytest
@pytest.mark.parametrize('state', np.array([[1, 0, 0, 0], [1, 0, 0, 1], [3, 5, 2, 7], [0.7823, 0.12323, 0.4312, 0.12321], [23, 43, 12, 19], [1j, 0, 0, 0], [1j, 0, 0, 1j], [1j, -1j, -1j, 1j], [1 + 1j, 0, 0, 0], [1 + 1j, 0, 1 + 1j, 0], [3 + 1j, 5 + 8j, 21, 0.85j]]))
def test_state_prep_channel_kraus(state):
    qubits = cirq.LineQubit.range(2)
    gate = cirq.StatePreparationChannel(state)(qubits[0], qubits[1])
    cirq.testing.assert_consistent_channel(gate)
    assert not cirq.has_mixture(gate)
    state = state / np.linalg.norm(state)
    np.testing.assert_almost_equal(cirq.kraus(gate), (np.array([state, np.zeros(4), np.zeros(4), np.zeros(4)]).T, np.array([np.zeros(4), state, np.zeros(4), np.zeros(4)]).T, np.array([np.zeros(4), np.zeros(4), state, np.zeros(4)]).T, np.array([np.zeros(4), np.zeros(4), np.zeros(4), state]).T))