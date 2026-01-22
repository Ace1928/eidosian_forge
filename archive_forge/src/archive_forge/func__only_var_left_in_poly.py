from .polynomial import Polynomial
from .fieldExtensions import my_rnfequation
from ..pari import pari
def _only_var_left_in_poly(poly, extension_vars):
    """
    Checks whether that there is only one other variable besides
    the variables in extension_vars.
    In other words, if the variables in extension_vars are bound,
    checks that the polynomial has only one free variable and returns
    its name.
    """
    vars_left = set(poly.variables()) - set(extension_vars)
    no_vars_left = len(vars_left)
    assert no_vars_left > 0
    if no_vars_left > 1:
        return None
    return list(vars_left)[0]