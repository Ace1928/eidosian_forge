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
def applyfunc(self, f):
    """Apply a function to each element of the N-dim array.

        Examples
        ========

        >>> from sympy import ImmutableDenseNDimArray
        >>> m = ImmutableDenseNDimArray([i*2+j for i in range(2) for j in range(2)], (2, 2))
        >>> m
        [[0, 1], [2, 3]]
        >>> m.applyfunc(lambda i: 2*i)
        [[0, 2], [4, 6]]
        """
    from sympy.tensor.array import SparseNDimArray
    from sympy.tensor.array.arrayop import Flatten
    if isinstance(self, SparseNDimArray) and f(S.Zero) == 0:
        return type(self)({k: f(v) for k, v in self._sparse_array.items() if f(v) != 0}, self.shape)
    return type(self)(map(f, Flatten(self)), self.shape)