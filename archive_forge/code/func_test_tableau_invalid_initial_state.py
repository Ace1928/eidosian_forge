import numpy as np
import pytest
import cirq
def test_tableau_invalid_initial_state():
    with pytest.raises(ValueError, match='2*num_qubits columns and of type bool.'):
        cirq.CliffordTableau(1, rs=np.zeros(1, dtype=bool))
    with pytest.raises(ValueError, match='2*num_qubits rows, num_qubits columns, and of type bool.'):
        cirq.CliffordTableau(1, xs=np.zeros(1, dtype=bool))
    with pytest.raises(ValueError, match='2*num_qubits rows, num_qubits columns, and of type bool.'):
        cirq.CliffordTableau(1, zs=np.zeros(1, dtype=bool))