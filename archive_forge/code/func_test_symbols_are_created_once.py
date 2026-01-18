import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_symbols_are_created_once():
    """
    Test that a symbol is cached and reused when it appears in an expression
    more than once.
    """
    expr = sy.Add(x, x, evaluate=False)
    comp = theano_code_(expr)
    assert theq(comp, xt + xt)
    assert not theq(comp, xt + theano_code_(x))