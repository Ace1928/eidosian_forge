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
def _join_lines(a):
    i = 0
    while i < len(a):
        x = a[i]
        xend = x[-1]
        xstart = x[0]
        hit = True
        while hit:
            hit = False
            for j in range(i + 1, len(a)):
                if j >= len(a):
                    break
                if a[j][0] == xend:
                    hit = True
                    x.extend(a[j][1:])
                    xend = x[-1]
                    a.pop(j)
                    continue
                if a[j][0] == xstart:
                    hit = True
                    a[i] = reversed(a[j][1:]) + x
                    x = a[i]
                    xstart = a[i][0]
                    a.pop(j)
                    continue
                if a[j][-1] == xend:
                    hit = True
                    x.extend(reversed(a[j][:-1]))
                    xend = x[-1]
                    a.pop(j)
                    continue
                if a[j][-1] == xstart:
                    hit = True
                    a[i] = a[j][:-1] + x
                    x = a[i]
                    xstart = x[0]
                    a.pop(j)
                    continue
        i += 1
    return a