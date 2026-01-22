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
class WildTensorHead(TensorHead):
    """
    A wild object that is used to create ``WildTensor`` instances

    Explanation
    ===========

    Examples
    ========
    >>> from sympy.tensor.tensor import TensorHead, TensorIndex, WildTensorHead, TensorIndexType
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex('p', R3)
    >>> q = TensorIndex('q', R3)

    A WildTensorHead can be created without specifying a ``TensorIndexType``

    >>> W = WildTensorHead("W")

    Calling it with a ``TensorIndex`` creates a ``WildTensor`` instance.

    >>> type(W(p))
    <class 'sympy.tensor.tensor.WildTensor'>

    The ``TensorIndexType`` is automatically detected from the index that is passed

    >>> W(p).component
    W(R3)

    Calling it with no indices returns an object that can match tensors with any number of indices.

    >>> K = TensorHead('K', [R3])
    >>> Q = TensorHead('Q', [R3, R3])
    >>> W().matches(K(p))
    {W: K(p)}
    >>> W().matches(Q(p,q))
    {W: Q(p, q)}

    If you want to ignore the order of indices while matching, pass ``unordered_indices=True``.

    >>> U = WildTensorHead("U", unordered_indices=True)
    >>> W(p,q).matches(Q(q,p))
    >>> U(p,q).matches(Q(q,p))
    {U(R3,R3): _WildTensExpr(Q(q, p))}

    Parameters
    ==========
    name : name of the tensor
    unordered_indices : whether the order of the indices matters for matching
        (default: False)

    See also
    ========
    ``WildTensor``
    ``TensorHead``

    """

    def __new__(cls, name, index_types=None, symmetry=None, comm=0, unordered_indices=False):
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        else:
            raise ValueError('invalid name')
        if index_types is None:
            index_types = []
        if symmetry is None:
            symmetry = TensorSymmetry.no_symmetry(len(index_types))
        else:
            assert symmetry.rank == len(index_types)
        if symmetry != TensorSymmetry.no_symmetry(len(index_types)):
            raise NotImplementedError('Wild matching based on symmetry is not implemented.')
        obj = Basic.__new__(cls, name_symbol, Tuple(*index_types), sympify(symmetry), sympify(comm), sympify(unordered_indices))
        obj.comm = TensorManager.comm_symbols2i(comm)
        obj.unordered_indices = unordered_indices
        return obj

    def __call__(self, *indices, **kwargs):
        tensor = WildTensor(self, indices, **kwargs)
        return tensor.doit()