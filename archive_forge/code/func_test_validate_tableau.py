import numpy as np
import pytest
import cirq
def test_validate_tableau():
    num_qubits = 4
    for i in range(2 ** num_qubits):
        t = cirq.CliffordTableau(initial_state=i, num_qubits=num_qubits)
        assert t._validate()
    t = cirq.CliffordTableau(num_qubits=2)
    assert t._validate()
    _H(t, 0)
    assert t._validate()
    _X(t, 0)
    assert t._validate()
    _Z(t, 1)
    assert t._validate()
    _CNOT(t, 0, 1)
    assert t._validate()
    _CNOT(t, 1, 0)
    assert t._validate()
    t.xs = np.zeros((4, 2))
    assert not t._validate()