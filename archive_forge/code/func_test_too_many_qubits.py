import itertools
import numpy as np
import pytest
import cirq
def test_too_many_qubits():
    a, b = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='single qubit'):
        _ = cirq.X.on(a, b)
    x = cirq.X(a)
    with pytest.raises(ValueError, match='len\\(new_qubits\\)'):
        _ = x.with_qubits(a, b)