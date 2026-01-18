from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('channel', (cirq.I, cirq.X, cirq.CNOT, cirq.depolarize(0.1), cirq.depolarize(0.1, n_qubits=2), cirq.amplitude_damp(0.2)))
def test_operation_to_choi(channel):
    """Verifies that cirq.operation_to_choi correctly computes the Choi matrix."""
    n_qubits = cirq.num_qubits(channel)
    actual = cirq.operation_to_choi(channel)
    expected = compute_choi(channel)
    assert np.isclose(np.trace(actual), 2 ** n_qubits)
    assert np.all(actual == expected)