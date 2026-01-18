import numpy as np
import pytest
import sympy
import cirq
def test_yy_pow():
    assert cirq.YY ** 0.5 != cirq.YY ** (-0.5)
    assert cirq.YY ** (-1) == cirq.YY
    assert (cirq.YY ** (-1)) ** 0.5 == cirq.YY ** (-0.5)