import numpy as np
import pytest
import cirq
def test_tableau_then_with_bad_input():
    t1 = cirq.CliffordTableau(1)
    t2 = cirq.CliffordTableau(2)
    with pytest.raises(ValueError, match='Mismatched number of qubits of two tableaux: 1 vs 2.'):
        t1.then(t2)
    with pytest.raises(TypeError):
        t1.then(cirq.X)