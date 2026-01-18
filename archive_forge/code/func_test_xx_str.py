import numpy as np
import pytest
import sympy
import cirq
def test_xx_str():
    assert str(cirq.XX) == 'XX'
    assert str(cirq.XX ** 0.5) == 'XX**0.5'
    assert str(cirq.XXPowGate(global_shift=0.1)) == 'XX'