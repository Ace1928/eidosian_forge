import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def theano_code_(expr, **kwargs):
    """ Wrapper for theano_code that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    with warns_deprecated_sympy():
        return theano_code(expr, **kwargs)