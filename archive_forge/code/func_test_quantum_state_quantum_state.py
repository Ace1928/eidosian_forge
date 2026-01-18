import numpy as np
import pytest
import cirq
import cirq.testing
def test_quantum_state_quantum_state():
    state_vector_1 = cirq.one_hot(shape=(4,), dtype=np.complex128)
    quantum_state = cirq.QuantumState(state_vector_1)
    state = cirq.quantum_state(quantum_state)
    assert state is quantum_state
    assert state.data is quantum_state.data
    assert state.dtype == np.complex128
    state = cirq.quantum_state(quantum_state, copy=True)
    assert state is not quantum_state
    assert state.data is not quantum_state.data
    assert state.dtype == np.complex128
    state = cirq.quantum_state(quantum_state, dtype=np.complex64)
    assert state is not quantum_state
    assert state.data is not quantum_state.data
    assert state.dtype == np.complex64
    with pytest.raises(ValueError, match='qid shape'):
        state = cirq.quantum_state(quantum_state, qid_shape=(4,))