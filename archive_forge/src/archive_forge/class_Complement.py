from typing import Any, Callable
from functools import reduce
from collections import defaultdict
import inspect
from sympy.core.kind import Kind, UndefinedKind, NumberKind
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import (FuzzyBool, fuzzy_bool, fuzzy_or, fuzzy_and,
from sympy.core.numbers import Float, Integer
from sympy.core.operations import LatticeOp
from sympy.core.parameters import global_parameters
from sympy.core.relational import Eq, Ne, is_lt
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import And, Or, Not, Xor, true, false
from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import (iproduct, sift, roundrobin, iterable,
from sympy.utilities.misc import func_name, filldedent
from mpmath import mpi, mpf
from mpmath.libmp.libmpf import prec_to_dps
class Complement(Set):
    """Represents the set difference or relative complement of a set with
    another set.

    $$A - B = \\{x \\in A \\mid x \\notin B\\}$$


    Examples
    ========

    >>> from sympy import Complement, FiniteSet
    >>> Complement(FiniteSet(0, 1, 2), FiniteSet(1))
    {0, 2}

    See Also
    =========

    Intersection, Union

    References
    ==========

    .. [1] https://mathworld.wolfram.com/ComplementSet.html
    """
    is_Complement = True

    def __new__(cls, a, b, evaluate=True):
        a, b = map(_sympify, (a, b))
        if evaluate:
            return Complement.reduce(a, b)
        return Basic.__new__(cls, a, b)

    @staticmethod
    def reduce(A, B):
        """
        Simplify a :class:`Complement`.

        """
        if B == S.UniversalSet or A.is_subset(B):
            return S.EmptySet
        if isinstance(B, Union):
            return Intersection(*(s.complement(A) for s in B.args))
        result = B._complement(A)
        if result is not None:
            return result
        else:
            return Complement(A, B, evaluate=False)

    def _contains(self, other):
        A = self.args[0]
        B = self.args[1]
        return And(A.contains(other), Not(B.contains(other)))

    def as_relational(self, symbol):
        """Rewrite a complement in terms of equalities and logic
        operators"""
        A, B = self.args
        A_rel = A.as_relational(symbol)
        B_rel = Not(B.as_relational(symbol))
        return And(A_rel, B_rel)

    def _kind(self):
        return self.args[0].kind

    @property
    def is_iterable(self):
        if self.args[0].is_iterable:
            return True

    @property
    def is_finite_set(self):
        A, B = self.args
        a_finite = A.is_finite_set
        if a_finite is True:
            return True
        elif a_finite is False and B.is_finite_set:
            return False

    def __iter__(self):
        A, B = self.args
        for a in A:
            if a not in B:
                yield a
            else:
                continue