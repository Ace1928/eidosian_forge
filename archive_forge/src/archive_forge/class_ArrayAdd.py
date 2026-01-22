import collections.abc
import operator
from collections import defaultdict, Counter
from functools import reduce
import itertools
from itertools import accumulate
from typing import Optional, List, Tuple as tTuple
import typing
from sympy.core.numbers import Integer
from sympy.core.relational import Equality
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import (Dummy, Symbol)
from sympy.matrices.common import MatrixCommon
from sympy.matrices.expressions.diagonal import diagonalize_vector
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensordiagonal, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.utils import _apply_recursively_over_nested_lists, _sort_contraction_indices, \
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.core.sympify import _sympify
class ArrayAdd(_CodegenArrayAbstract):
    """
    Class for elementwise array additions.
    """

    def __new__(cls, *args, **kwargs):
        args = [_sympify(arg) for arg in args]
        ranks = [get_rank(arg) for arg in args]
        ranks = list(set(ranks))
        if len(ranks) != 1:
            raise ValueError('summing arrays of different ranks')
        shapes = [arg.shape for arg in args]
        if len({i for i in shapes if i is not None}) > 1:
            raise ValueError('mismatching shapes in addition')
        canonicalize = kwargs.pop('canonicalize', False)
        obj = Basic.__new__(cls, *args)
        obj._subranks = ranks
        if any((i is None for i in shapes)):
            obj._shape = None
        else:
            obj._shape = shapes[0]
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        args = self.args
        args = self._flatten_args(args)
        shapes = [get_shape(arg) for arg in args]
        args = [arg for arg in args if not isinstance(arg, (ZeroArray, ZeroMatrix))]
        if len(args) == 0:
            if any((i for i in shapes if i is None)):
                raise NotImplementedError('cannot handle addition of ZeroMatrix/ZeroArray and undefined shape object')
            return ZeroArray(*shapes[0])
        elif len(args) == 1:
            return args[0]
        return self.func(*args, canonicalize=False)

    @classmethod
    def _flatten_args(cls, args):
        new_args = []
        for arg in args:
            if isinstance(arg, ArrayAdd):
                new_args.extend(arg.args)
            else:
                new_args.append(arg)
        return new_args

    def as_explicit(self):
        return reduce(operator.add, [arg.as_explicit() if hasattr(arg, 'as_explicit') else arg for arg in self.args])