import re
from typing import List, Callable
from sympy.core.sympify import _sympify
from sympy.external import import_module
from sympy.functions import (log, sin, cos, tan, cot, csc, sec, erf, gamma, uppergamma)
from sympy.functions.elementary.hyperbolic import acosh, asinh, atanh, acoth, acsch, asech, cosh, sinh, tanh, coth, sech, csch
from sympy.functions.elementary.trigonometric import atan, acsc, asin, acot, acos, asec
from sympy.functions.special.error_functions import fresnelc, fresnels, erfc, erfi, Ei
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import (Equality, Unequality)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.integrals.integrals import Integral
from sympy.printing.repr import srepr
from sympy.utilities.decorator import doctest_depends_on
def _get_custom_constraint(self, constraint_expr: Expr, condition_template: str) -> Callable[..., Expr]:
    wilds = [x.name for x in constraint_expr.atoms(_WildAbstract)]
    lambdaargs = ', '.join(wilds)
    fullexpr = _get_srepr(constraint_expr)
    condition = condition_template.format(fullexpr)
    return matchpy.CustomConstraint(self._get_lambda(f'lambda {lambdaargs}: ({condition})'))