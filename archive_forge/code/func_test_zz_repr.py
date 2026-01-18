import numpy as np
import pytest
import sympy
import cirq
def test_zz_repr():
    assert repr(cirq.ZZPowGate()) == 'cirq.ZZ'
    assert repr(cirq.ZZPowGate(exponent=0.5)) == '(cirq.ZZ**0.5)'