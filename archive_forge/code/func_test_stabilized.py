import numpy as np
import pytest
import cirq
def test_stabilized():
    for state in cirq.PAULI_STATES:
        val, gate = state.stabilized_by()
        matrix = cirq.unitary(gate)
        vec = state.state_vector()
        np.testing.assert_allclose(matrix @ vec, val * vec)