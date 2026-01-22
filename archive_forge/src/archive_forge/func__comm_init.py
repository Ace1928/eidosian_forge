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
def _comm_init(self):
    self._comm = [{} for i in range(3)]
    for i in range(3):
        self._comm[0][i] = 0
        self._comm[i][0] = 0
    self._comm[1][1] = 1
    self._comm[2][1] = None
    self._comm[1][2] = None
    self._comm_symbols2i = {0: 0, 1: 1, 2: 2}
    self._comm_i2symbol = {0: 0, 1: 1, 2: 2}