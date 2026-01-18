import numpy as np
import pytest
import sympy
import cirq
def test_xx_repr():
    assert repr(cirq.XXPowGate()) == 'cirq.XX'
    assert repr(cirq.XXPowGate(exponent=0.5)) == '(cirq.XX**0.5)'