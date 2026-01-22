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
@classmethod
def _check_permutation_mapping(cls, expr, permutation):
    subranks = expr.subranks
    index2arg = [i for i, arg in enumerate(expr.args) for j in range(expr.subranks[i])]
    permuted_indices = [permutation(i) for i in range(expr.subrank())]
    new_args = list(expr.args)
    arg_candidate_index = index2arg[permuted_indices[0]]
    current_indices = []
    new_permutation = []
    inserted_arg_cand_indices = set()
    for i, idx in enumerate(permuted_indices):
        if index2arg[idx] != arg_candidate_index:
            new_permutation.extend(current_indices)
            current_indices = []
            arg_candidate_index = index2arg[idx]
        current_indices.append(idx)
        arg_candidate_rank = subranks[arg_candidate_index]
        if len(current_indices) == arg_candidate_rank:
            new_permutation.extend(sorted(current_indices))
            local_current_indices = [j - min(current_indices) for j in current_indices]
            i1 = index2arg[i]
            new_args[i1] = _permute_dims(new_args[i1], Permutation(local_current_indices))
            inserted_arg_cand_indices.add(arg_candidate_index)
            current_indices = []
    new_permutation.extend(current_indices)
    args_positions = list(range(len(new_args)))
    maps = {}
    cumulative_subranks = [0] + list(accumulate(subranks))
    for i in range(len(subranks)):
        s = {index2arg[new_permutation[j]] for j in range(cumulative_subranks[i], cumulative_subranks[i + 1])}
        if len(s) != 1:
            continue
        elem = next(iter(s))
        if i != elem:
            maps[i] = elem
    lines = []
    current_line = []
    while maps:
        if len(current_line) == 0:
            k, v = maps.popitem()
            current_line.append(k)
        else:
            k = current_line[-1]
            if k not in maps:
                current_line = []
                continue
            v = maps.pop(k)
        if v in current_line:
            lines.append(current_line)
            current_line = []
            continue
        current_line.append(v)
    for line in lines:
        for i, e in enumerate(line):
            args_positions[line[(i + 1) % len(line)]] = e
    permutation_blocks = [[new_permutation[cumulative_subranks[i] + j] for j in range(e)] for i, e in enumerate(subranks)]
    new_args = [new_args[i] for i in args_positions]
    new_permutation_blocks = [permutation_blocks[i] for i in args_positions]
    new_permutation2 = [j for i in new_permutation_blocks for j in i]
    return (_array_tensor_product(*new_args), Permutation(new_permutation2))