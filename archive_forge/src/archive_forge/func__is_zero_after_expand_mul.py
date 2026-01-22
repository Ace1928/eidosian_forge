from contextlib import contextmanager
from threading import local
from sympy.core.function import expand_mul
def _is_zero_after_expand_mul(x):
    """Tests by expand_mul only, suitable for polynomials and rational
    functions."""
    return expand_mul(x) == 0