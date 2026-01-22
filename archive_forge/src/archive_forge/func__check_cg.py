from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j
from sympy.printing.precedence import PRECEDENCE
def _check_cg(cg_term, expr, length, sign=None):
    """Checks whether a term matches the given expression"""
    matches = cg_term.match(expr)
    if matches is None:
        return
    if sign is not None:
        if not isinstance(sign, tuple):
            raise TypeError('sign must be a tuple')
        if not sign[0] == sign[1].subs(matches):
            return
    if len(matches) == length:
        return matches