import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_unitary_construction():
    with pytest.raises(TypeError):
        _ = cirq.ApplyUnitaryArgs.for_unitary()
    np.testing.assert_allclose(cirq.ApplyUnitaryArgs.for_unitary(num_qubits=3).target_tensor, cirq.eye_tensor((2,) * 3, dtype=np.complex128))