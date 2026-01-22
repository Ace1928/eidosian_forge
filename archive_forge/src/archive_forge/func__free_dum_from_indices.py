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
def _free_dum_from_indices(*indices):
    """
        Convert ``indices`` into ``free``, ``dum`` for single component tensor.

        Explanation
        ===========

        ``free``     list of tuples ``(index, pos, 0)``,
                     where ``pos`` is the position of index in
                     the list of indices formed by the component tensors

        ``dum``      list of tuples ``(pos_contr, pos_cov, 0, 0)``

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices,             _IndexStructure
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2, m3 = tensor_indices('m0,m1,m2,m3', Lorentz)
        >>> _IndexStructure._free_dum_from_indices(m0, m1, -m1, m3)
        ([(m0, 0), (m3, 3)], [(1, 2)])
        """
    n = len(indices)
    if n == 1:
        return ([(indices[0], 0)], [])
    free = [True] * len(indices)
    index_dict = {}
    dum = []
    for i, index in enumerate(indices):
        name = index.name
        typ = index.tensor_index_type
        contr = index.is_up
        if (name, typ) in index_dict:
            is_contr, pos = index_dict[name, typ]
            if is_contr:
                if contr:
                    raise ValueError('two equal contravariant indices in slots %d and %d' % (pos, i))
                else:
                    free[pos] = False
                    free[i] = False
            elif contr:
                free[pos] = False
                free[i] = False
            else:
                raise ValueError('two equal covariant indices in slots %d and %d' % (pos, i))
            if contr:
                dum.append((i, pos))
            else:
                dum.append((pos, i))
        else:
            index_dict[name, typ] = (index.is_up, i)
    free = [(index, i) for i, index in enumerate(indices) if free[i]]
    free.sort()
    return (free, dum)