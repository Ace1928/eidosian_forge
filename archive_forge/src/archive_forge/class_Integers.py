from functools import reduce
from itertools import product
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import fuzzy_not, fuzzy_or, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.numbers import oo, igcd, Rational
from sympy.core.relational import Eq, is_eq
from sympy.core.kind import NumberKind
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.logic.boolalg import And, Or
from .sets import Set, Interval, Union, FiniteSet, ProductSet, SetKind
from sympy.utilities.misc import filldedent
class Integers(Set, metaclass=Singleton):
    """
    Represents all integers: positive, negative and zero. This set is also
    available as the singleton ``S.Integers``.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Integers)
    >>> next(iterable)
    0
    >>> next(iterable)
    1
    >>> next(iterable)
    -1
    >>> next(iterable)
    2

    >>> pprint(S.Integers.intersect(Interval(-4, 4)))
    {-4, -3, ..., 4}

    See Also
    ========

    Naturals0 : non-negative integers
    Integers : positive and negative integers and zero
    """
    is_iterable = True
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        return other.is_integer

    def __iter__(self):
        yield S.Zero
        i = S.One
        while True:
            yield i
            yield (-i)
            i = i + 1

    @property
    def _inf(self):
        return S.NegativeInfinity

    @property
    def _sup(self):
        return S.Infinity

    @property
    def _boundary(self):
        return self

    def _kind(self):
        return SetKind(NumberKind)

    def as_relational(self, x):
        return And(Eq(floor(x), x), -oo < x, x < oo)

    def _eval_is_subset(self, other):
        return Range(-oo, oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(-oo, oo).is_superset(other)