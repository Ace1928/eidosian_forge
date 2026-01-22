from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod
from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools
from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.containers import Tuple, Dict
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift
def _sort_args_for_sorted_components(self):
    """
        Returns the ``args`` sorted according to the components commutation
        properties.

        Explanation
        ===========

        The sorting is done taking into account the commutation group
        of the component tensors.
        """
    cv = [arg for arg in self.args if isinstance(arg, TensExpr)]
    sign = 1
    n = len(cv) - 1
    for i in range(n):
        for j in range(n, i, -1):
            c = cv[j - 1].commutes_with(cv[j])
            if c not in (0, 1):
                continue
            typ1 = sorted(set(cv[j - 1].component.index_types), key=lambda x: x.name)
            typ2 = sorted(set(cv[j].component.index_types), key=lambda x: x.name)
            if (typ1, cv[j - 1].component.name) > (typ2, cv[j].component.name):
                cv[j - 1], cv[j] = (cv[j], cv[j - 1])
                if c:
                    sign = -sign
    coeff = sign * self.coeff
    if coeff != 1:
        return [coeff] + cv
    return cv