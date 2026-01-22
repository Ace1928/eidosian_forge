from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.printing.defaults import Printable
import itertools
from collections.abc import Iterable
def _slice_expand(self, s, dim):
    if not isinstance(s, slice):
        return (s,)
    start, stop, step = s.indices(dim)
    return [start + i * step for i in range((stop - start) // step)]