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
def _check_permutations_on_data(self, tens, data):
    from .array import permutedims
    from .array.arrayop import Flatten
    if isinstance(tens, TensorHead):
        rank = tens.rank
        generators = tens.symmetry.generators
    elif isinstance(tens, Tensor):
        rank = tens.rank
        generators = tens.components[0].symmetry.generators
    elif isinstance(tens, TensorIndexType):
        rank = tens.metric.rank
        generators = tens.metric.symmetry.generators
    for gener in generators:
        sign_change = +1 if gener(rank) == rank else -1
        data_swapped = data
        last_data = data
        permute_axes = list(map(gener, range(rank)))
        for i in range(gener.order() - 1):
            data_swapped = permutedims(data_swapped, permute_axes)
            if any(Flatten(last_data - sign_change * data_swapped)):
                raise ValueError('Component data symmetry structure error')
            last_data = data_swapped