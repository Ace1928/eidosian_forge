import numpy as np
import pytest
import sympy
import cirq
def test_zz_str():
    assert str(cirq.ZZ) == 'ZZ'
    assert str(cirq.ZZ ** 0.5) == 'ZZ**0.5'
    assert str(cirq.ZZPowGate(global_shift=0.1)) == 'ZZ'
    iZZ = cirq.ZZPowGate(global_shift=0.5)
    assert str(iZZ) == 'ZZ'