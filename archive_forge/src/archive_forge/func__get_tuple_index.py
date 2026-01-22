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
def _get_tuple_index(self, integer_index):
    index = []
    for i, sh in enumerate(reversed(self.shape)):
        index.append(integer_index % sh)
        integer_index //= sh
    index.reverse()
    return tuple(index)