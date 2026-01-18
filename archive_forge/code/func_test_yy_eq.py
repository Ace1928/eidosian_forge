import numpy as np
import pytest
import sympy
import cirq
def test_yy_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.YY, cirq.YYPowGate(), cirq.YYPowGate(exponent=1, global_shift=0), cirq.YYPowGate(exponent=3, global_shift=0))
    eq.add_equality_group(cirq.YY ** 0.5, cirq.YY ** 2.5, cirq.YY ** 4.5)
    eq.add_equality_group(cirq.YY ** 0.25, cirq.YY ** 2.25, cirq.YY ** (-1.75))
    iYY = cirq.YYPowGate(global_shift=0.5)
    eq.add_equality_group(iYY ** 0.5, iYY ** 4.5)
    eq.add_equality_group(iYY ** 2.5, iYY ** 6.5)