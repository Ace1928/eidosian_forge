import numpy as np
import pytest
import sympy
import cirq
def test_yy_str():
    assert str(cirq.YY) == 'YY'
    assert str(cirq.YY ** 0.5) == 'YY**0.5'
    assert str(cirq.YYPowGate(global_shift=0.1)) == 'YY'
    iYY = cirq.YYPowGate(global_shift=0.5)
    assert str(iYY) == 'YY'