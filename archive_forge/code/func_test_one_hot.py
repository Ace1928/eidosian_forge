import numbers
from typing import List
import numpy as np
import pytest
import sympy
import cirq
from cirq.ops.dense_pauli_string import _vectorized_pauli_mul_phase
def test_one_hot():
    f = cirq.DensePauliString
    m = cirq.MutableDensePauliString
    assert cirq.DensePauliString.one_hot(index=3, length=5, pauli=cirq.X) == f('IIIXI')
    assert cirq.MutableDensePauliString.one_hot(index=3, length=5, pauli=cirq.X) == m('IIIXI')
    assert cirq.BaseDensePauliString.one_hot(index=0, length=5, pauli='X') == f('XIIII')
    assert cirq.BaseDensePauliString.one_hot(index=0, length=5, pauli='Y') == f('YIIII')
    assert cirq.BaseDensePauliString.one_hot(index=0, length=5, pauli='Z') == f('ZIIII')
    assert cirq.BaseDensePauliString.one_hot(index=0, length=5, pauli='I') == f('IIIII')
    assert cirq.BaseDensePauliString.one_hot(index=0, length=5, pauli=cirq.X) == f('XIIII')
    assert cirq.BaseDensePauliString.one_hot(index=0, length=5, pauli=cirq.Y) == f('YIIII')
    assert cirq.BaseDensePauliString.one_hot(index=0, length=5, pauli=cirq.Z) == f('ZIIII')
    assert cirq.BaseDensePauliString.one_hot(index=0, length=5, pauli=cirq.I) == f('IIIII')
    with pytest.raises(IndexError):
        _ = cirq.BaseDensePauliString.one_hot(index=50, length=5, pauli=cirq.X)
    with pytest.raises(IndexError):
        _ = cirq.BaseDensePauliString.one_hot(index=0, length=0, pauli=cirq.X)