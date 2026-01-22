import itertools
from sympy.core import S
from sympy.core.sorting import default_sort_key
from sympy.polys import Poly, groebner, roots
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.polys.polyerrors import (ComputationFailed,
from sympy.simplify import rcollect
from sympy.utilities import postfixes
from sympy.utilities.misc import filldedent
def _is_univariate(f):
    """Returns True if 'f' is univariate in its last variable. """
    for monom in f.monoms():
        if any(monom[:-1]):
            return False
    return True