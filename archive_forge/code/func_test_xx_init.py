import numpy as np
import pytest
import sympy
import cirq
def test_xx_init():
    assert cirq.XXPowGate(exponent=1).exponent == 1
    v = cirq.XXPowGate(exponent=0.5)
    assert v.exponent == 0.5