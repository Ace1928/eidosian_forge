import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_strat_apply_unitary_from_decompose():
    state = np.eye(2, dtype=np.complex128)
    args = cirq.ApplyUnitaryArgs(target_tensor=state, available_buffer=np.zeros_like(state), axes=(0,))
    np.testing.assert_allclose(cirq.apply_unitaries([DecomposableGate(cirq.X, False)(cirq.LineQubit(0))], [cirq.LineQubit(0)], args), [[0, 1], [1, 0]])
    with pytest.raises(TypeError):
        _ = cirq.apply_unitaries([DecomposableGate(NotDecomposableGate(), True)(cirq.LineQubit(0))], [cirq.LineQubit(0)], args)