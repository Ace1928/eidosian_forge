import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP
from sympy.utilities.exceptions import ignore_warnings
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.aesaracode import (aesara_code, dim_handling,
def aesara_function_(inputs, outputs, **kwargs):
    """ Wrapper for aesara_function that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    return aesara_function(inputs, outputs, **kwargs)