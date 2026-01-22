import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP
from sympy.utilities.exceptions import ignore_warnings
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.aesaracode import (aesara_code, dim_handling,
def aesara_code_(expr, **kwargs):
    """ Wrapper for aesara_code that uses a new, empty cache by default. """
    kwargs.setdefault('cache', {})
    return aesara_code(expr, **kwargs)