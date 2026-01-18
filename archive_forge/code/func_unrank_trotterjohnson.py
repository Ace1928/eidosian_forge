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
@classmethod
def unrank_trotterjohnson(cls, size, rank):
    """
        Trotter Johnson permutation unranking. See [4] section 2.4.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> Permutation.unrank_trotterjohnson(5, 10)
        Permutation([0, 3, 1, 2, 4])

        See Also
        ========

        rank_trotterjohnson, next_trotterjohnson
        """
    perm = [0] * size
    r2 = 0
    n = ifac(size)
    pj = 1
    for j in range(2, size + 1):
        pj *= j
        r1 = rank * pj // n
        k = r1 - j * r2
        if r2 % 2 == 0:
            for i in range(j - 1, j - k - 1, -1):
                perm[i] = perm[i - 1]
            perm[j - k - 1] = j - 1
        else:
            for i in range(j - 1, k, -1):
                perm[i] = perm[i - 1]
            perm[k] = j - 1
        r2 = r1
    return cls._af_new(perm)