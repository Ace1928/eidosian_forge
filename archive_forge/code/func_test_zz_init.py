import numpy as np
import pytest
import sympy
import cirq
def test_zz_init():
    assert cirq.ZZPowGate(exponent=1).exponent == 1
    v = cirq.ZZPowGate(exponent=0.5)
    assert v.exponent == 0.5