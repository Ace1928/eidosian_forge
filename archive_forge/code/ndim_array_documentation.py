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

        NDimArray instances can be compared to each other.
        Instances equal if they have same shape and data.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(2, 3)
        >>> b = MutableDenseNDimArray.zeros(2, 3)
        >>> a == b
        True
        >>> c = a.reshape(3, 2)
        >>> c == b
        False
        >>> a[0,0] = 1
        >>> b[0,0] = 2
        >>> a == b
        False
        