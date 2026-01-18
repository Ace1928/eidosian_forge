import pytest
import sympy
import cirq
def test_symbol_special_case():
    x = sympy.Symbol('x')
    assert cirq.mul(x, 1.0) is x
    assert cirq.mul(1.0, x) is x
    assert str(cirq.mul(-1.0, x)) == '-x'
    assert str(cirq.mul(x, -1.0)) == '-x'