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
def transpositions(self):
    """
        Return the permutation decomposed into a list of transpositions.

        Explanation
        ===========

        It is always possible to express a permutation as the product of
        transpositions, see [1]

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([[1, 2, 3], [0, 4, 5, 6, 7]])
        >>> t = p.transpositions()
        >>> t
        [(0, 7), (0, 6), (0, 5), (0, 4), (1, 3), (1, 2)]
        >>> print(''.join(str(c) for c in t))
        (0, 7)(0, 6)(0, 5)(0, 4)(1, 3)(1, 2)
        >>> Permutation.rmul(*[Permutation([ti], size=p.size) for ti in t]) == p
        True

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Transposition_%28mathematics%29#Properties

        """
    a = self.cyclic_form
    res = []
    for x in a:
        nx = len(x)
        if nx == 2:
            res.append(tuple(x))
        elif nx > 2:
            first = x[0]
            for y in x[nx - 1:0:-1]:
                res.append((first, y))
    return res