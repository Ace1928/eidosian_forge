from __future__ import annotations
from typing import Any
from functools import reduce
from itertools import permutations
from sympy.combinatorics import Permutation
from sympy.core import (
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol, Dummy
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions import factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix
from sympy.solvers import solve
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.simplify.simplify import simplify
@classmethod
@cacheit
def _indirect_transformation(cls, sys1, sys2):
    rel = sys1.relations
    path = cls._dijkstra(sys1, sys2)
    transforms = []
    for s1, s2 in zip(path, path[1:]):
        if (s1, s2) in rel:
            transforms.append(rel[s1, s2])
        else:
            sym2, inv_exprs = rel[s2, s1]
            sym1 = tuple((Dummy() for i in sym2))
            ret = cls._solve_inverse(sym2, sym1, inv_exprs, s2, s1)
            ret = tuple((ret[s] for s in sym2))
            transforms.append((sym1, ret))
    syms = sys1.args[2]
    exprs = syms
    for newsyms, newexprs in transforms:
        exprs = tuple((e.subs(zip(newsyms, exprs)) for e in newexprs))
    return exprs