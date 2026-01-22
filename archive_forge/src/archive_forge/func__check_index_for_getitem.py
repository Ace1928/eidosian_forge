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
def _check_index_for_getitem(self, index):
    if isinstance(index, (SYMPY_INTS, Integer, slice)):
        index = (index,)
    if len(index) < self.rank():
        index = tuple(index) + tuple((slice(None) for i in range(len(index), self.rank())))
    if len(index) > self.rank():
        raise ValueError('Dimension of index greater than rank of array')
    return index