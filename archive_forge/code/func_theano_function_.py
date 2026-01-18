import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def theano_function_(inputs, outputs, **kwargs):
    """ Wrapper for theano_function that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    with warns_deprecated_sympy():
        return theano_function(inputs, outputs, **kwargs)