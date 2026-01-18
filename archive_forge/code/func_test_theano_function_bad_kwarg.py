import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_theano_function_bad_kwarg():
    """
    Passing an unknown keyword argument to theano_function() should raise an
    exception.
    """
    raises(Exception, lambda: theano_function_([x], [x + 1], foobar=3))