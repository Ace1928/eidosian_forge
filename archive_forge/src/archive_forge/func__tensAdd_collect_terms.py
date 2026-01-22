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
@staticmethod
def _tensAdd_collect_terms(args):
    terms_dict = defaultdict(list)
    scalars = S.Zero
    if isinstance(args[0], TensExpr):
        free_indices = set(args[0].get_free_indices())
    else:
        free_indices = set()
    for arg in args:
        if not isinstance(arg, TensExpr):
            if free_indices != set():
                raise ValueError('wrong valence')
            scalars += arg
            continue
        if free_indices != set(arg.get_free_indices()):
            raise ValueError('wrong valence')
        terms_dict[arg.nocoeff].append(arg.coeff)
    new_args = [TensMul(Add(*coeff), t).doit() for t, coeff in terms_dict.items() if Add(*coeff) != 0]
    if isinstance(scalars, Add):
        new_args = list(scalars.args) + new_args
    elif scalars != 0:
        new_args = [scalars] + new_args
    return new_args