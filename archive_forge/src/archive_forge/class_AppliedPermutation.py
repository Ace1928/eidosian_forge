import random
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from sympy.core.parameters import global_parameters
from sympy.core.basic import Atom
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.sympify import _sympify
from sympy.matrices import zeros
from sympy.polys.polytools import lcm
from sympy.printing.repr import srepr
from sympy.utilities.iterables import (flatten, has_variety, minlex,
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import ifac
from sympy.multipledispatch import dispatch
class AppliedPermutation(Expr):
    """A permutation applied to a symbolic variable.

    Parameters
    ==========

    perm : Permutation
    x : Expr

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.combinatorics import Permutation

    Creating a symbolic permutation function application:

    >>> x = Symbol('x')
    >>> p = Permutation(0, 1, 2)
    >>> p.apply(x)
    AppliedPermutation((0 1 2), x)
    >>> _.subs(x, 1)
    2
    """

    def __new__(cls, perm, x, evaluate=None):
        if evaluate is None:
            evaluate = global_parameters.evaluate
        perm = _sympify(perm)
        x = _sympify(x)
        if not isinstance(perm, Permutation):
            raise ValueError('{} must be a Permutation instance.'.format(perm))
        if evaluate:
            if x.is_Integer:
                return perm.apply(x)
        obj = super().__new__(cls, perm, x)
        return obj