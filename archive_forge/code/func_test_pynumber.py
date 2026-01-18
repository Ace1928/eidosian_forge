from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_pynumber():
    a = sympy.FF(7)(3)
    b = sympify(a)
    assert isinstance(b, PyNumber)
    a = a + 1
    b = b + 1
    assert isinstance(b, PyNumber)
    assert b == a
    assert a == b
    assert str(a) == str(b)
    a = 1 - a
    b = 1 - b
    assert isinstance(b, PyNumber)
    assert b == a
    assert a == b
    a = 2 * a
    b = 2 * b
    assert isinstance(b, PyNumber)
    assert b == a
    assert a == b
    if sympy.__version__ != '1.2':
        a = 2 / a
        b = 2 / b
        assert isinstance(b, PyNumber)
        assert b == a
        assert a == b
    x = Symbol('x')
    b = x * sympy.FF(7)(3)
    assert isinstance(b, Mul)
    b = b / x
    assert isinstance(b, PyNumber)