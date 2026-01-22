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
class ArrayTensorProduct(_CodegenArrayAbstract):
    """
    Class to represent the tensor product of array-like objects.
    """

    def __new__(cls, *args, **kwargs):
        args = [_sympify(arg) for arg in args]
        canonicalize = kwargs.pop('canonicalize', False)
        ranks = [get_rank(arg) for arg in args]
        obj = Basic.__new__(cls, *args)
        obj._subranks = ranks
        shapes = [get_shape(i) for i in args]
        if any((i is None for i in shapes)):
            obj._shape = None
        else:
            obj._shape = tuple((j for i in shapes for j in i))
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        args = self.args
        args = self._flatten(args)
        ranks = [get_rank(arg) for arg in args]
        permutation_cycles = []
        for i, arg in enumerate(args):
            if not isinstance(arg, PermuteDims):
                continue
            permutation_cycles.extend([[k + sum(ranks[:i]) for k in j] for j in arg.permutation.cyclic_form])
            args[i] = arg.expr
        if permutation_cycles:
            return _permute_dims(_array_tensor_product(*args), Permutation(sum(ranks) - 1) * Permutation(permutation_cycles))
        if len(args) == 1:
            return args[0]
        if any((isinstance(arg, (ZeroArray, ZeroMatrix)) for arg in args)):
            shapes = reduce(operator.add, [get_shape(i) for i in args], ())
            return ZeroArray(*shapes)
        contractions = {i: arg for i, arg in enumerate(args) if isinstance(arg, ArrayContraction)}
        if contractions:
            ranks = [_get_subrank(arg) if isinstance(arg, ArrayContraction) else get_rank(arg) for arg in args]
            cumulative_ranks = list(accumulate([0] + ranks))[:-1]
            tp = _array_tensor_product(*[arg.expr if isinstance(arg, ArrayContraction) else arg for arg in args])
            contraction_indices = [tuple((cumulative_ranks[i] + k for k in j)) for i, arg in contractions.items() for j in arg.contraction_indices]
            return _array_contraction(tp, *contraction_indices)
        diagonals = {i: arg for i, arg in enumerate(args) if isinstance(arg, ArrayDiagonal)}
        if diagonals:
            inverse_permutation = []
            last_perm = []
            ranks = [get_rank(arg) for arg in args]
            cumulative_ranks = list(accumulate([0] + ranks))[:-1]
            for i, arg in enumerate(args):
                if isinstance(arg, ArrayDiagonal):
                    i1 = get_rank(arg) - len(arg.diagonal_indices)
                    i2 = len(arg.diagonal_indices)
                    inverse_permutation.extend([cumulative_ranks[i] + j for j in range(i1)])
                    last_perm.extend([cumulative_ranks[i] + j for j in range(i1, i1 + i2)])
                else:
                    inverse_permutation.extend([cumulative_ranks[i] + j for j in range(get_rank(arg))])
            inverse_permutation.extend(last_perm)
            tp = _array_tensor_product(*[arg.expr if isinstance(arg, ArrayDiagonal) else arg for arg in args])
            ranks2 = [_get_subrank(arg) if isinstance(arg, ArrayDiagonal) else get_rank(arg) for arg in args]
            cumulative_ranks2 = list(accumulate([0] + ranks2))[:-1]
            diagonal_indices = [tuple((cumulative_ranks2[i] + k for k in j)) for i, arg in diagonals.items() for j in arg.diagonal_indices]
            return _permute_dims(_array_diagonal(tp, *diagonal_indices), _af_invert(inverse_permutation))
        return self.func(*args, canonicalize=False)

    @classmethod
    def _flatten(cls, args):
        args = [i for arg in args for i in (arg.args if isinstance(arg, cls) else [arg])]
        return args

    def as_explicit(self):
        return tensorproduct(*[arg.as_explicit() if hasattr(arg, 'as_explicit') else arg for arg in self.args])