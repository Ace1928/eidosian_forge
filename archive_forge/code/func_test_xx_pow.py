import numpy as np
import pytest
import sympy
import cirq
def test_xx_pow():
    assert cirq.XX ** 0.5 != cirq.XX ** (-0.5)
    assert cirq.XX ** (-1) == cirq.XX
    assert (cirq.XX ** (-1)) ** 0.5 == cirq.XX ** (-0.5)