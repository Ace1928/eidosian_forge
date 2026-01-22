from sympy.core.relational import Eq, is_eq
from sympy.core.basic import Basic
from sympy.core.logic import fuzzy_and, fuzzy_bool
from sympy.logic.boolalg import And
from sympy.multipledispatch import dispatch
from sympy.sets.sets import tfn, ProductSet, Interval, FiniteSet, Set
@dispatch(Set, Set)
def _eval_is_eq(lhs, rhs):
    return tfn[fuzzy_and((a.is_subset(b) for a, b in [(lhs, rhs), (rhs, lhs)]))]