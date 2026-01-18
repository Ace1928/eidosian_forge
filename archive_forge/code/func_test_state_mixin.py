import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_state_mixin():

    class TestClass(cirq.StateVectorMixin):

        def state_vector(self, copy: Optional[bool]=None) -> np.ndarray:
            return np.array([0, 0, 1, 0])
    qubits = cirq.LineQubit.range(2)
    test = TestClass(qubit_map={qubits[i]: i for i in range(2)})
    assert test.dirac_notation() == '|10⟩'
    np.testing.assert_almost_equal(test.bloch_vector_of(qubits[0]), np.array([0, 0, -1]))
    np.testing.assert_almost_equal(test.density_matrix_of(qubits[0:1]), np.array([[0, 0], [0, 1]]))
    assert cirq.qid_shape(TestClass({qubits[i]: 1 - i for i in range(2)})) == (2, 2)
    assert cirq.qid_shape(TestClass({cirq.LineQid(i, i + 1): 2 - i for i in range(3)})) == (3, 2, 1)
    assert cirq.qid_shape(TestClass(), 'no shape') == 'no shape'
    with pytest.raises(ValueError, match='Qubit index out of bounds'):
        _ = TestClass({qubits[0]: 1})
    with pytest.raises(ValueError, match='Duplicate qubit index'):
        _ = TestClass({qubits[0]: 0, qubits[1]: 0})
    with pytest.raises(ValueError, match='Duplicate qubit index'):
        _ = TestClass({qubits[0]: 1, qubits[1]: 1})
    with pytest.raises(ValueError, match='Duplicate qubit index'):
        _ = TestClass({qubits[0]: -1, qubits[1]: 1})