from collections.abc import Iterable
from sympy.core.numbers import Number
from sympy.core.assumptions import StdFactKB
from sympy.core import Expr, Tuple, sympify, S
from sympy.core.symbol import _filter_assumptions, Symbol
from sympy.core.logic import fuzzy_bool, fuzzy_not
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.multipledispatch import dispatch
from sympy.utilities.iterables import is_sequence, NotIterable
from sympy.utilities.misc import filldedent
@property
def _diff_wrt(self):
    """Allow derivatives with respect to an ``Indexed`` object."""
    return True