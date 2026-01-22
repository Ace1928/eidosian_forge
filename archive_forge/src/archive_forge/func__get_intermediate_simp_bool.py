from contextlib import contextmanager
from threading import local
from sympy.core.function import expand_mul
def _get_intermediate_simp_bool(default=False, dotprodsimp=None):
    """Same as ``_get_intermediate_simp`` but returns bools instead of functions
    by default."""
    return _get_intermediate_simp(default, False, True, dotprodsimp)