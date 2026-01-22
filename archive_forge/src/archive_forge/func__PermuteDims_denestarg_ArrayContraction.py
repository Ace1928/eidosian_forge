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
def _PermuteDims_denestarg_ArrayContraction(cls, expr, permutation):
    if not isinstance(expr, ArrayContraction):
        return (expr, permutation)
    if not isinstance(expr.expr, ArrayTensorProduct):
        return (expr, permutation)
    args = expr.expr.args
    subranks = [get_rank(arg) for arg in expr.expr.args]
    contraction_indices = expr.contraction_indices
    contraction_indices_flat = [j for i in contraction_indices for j in i]
    cumul = list(accumulate([0] + subranks))
    permutation_array_blocks_up = []
    image_form = _af_invert(permutation.array_form)
    counter = 0
    for i, e in enumerate(subranks):
        current = []
        for j in range(cumul[i], cumul[i + 1]):
            if j in contraction_indices_flat:
                continue
            current.append(image_form[counter])
            counter += 1
        permutation_array_blocks_up.append(current)
    index_blocks = [list(range(cumul[i], cumul[i + 1])) for i, e in enumerate(expr.subranks)]
    index_blocks_up = expr._push_indices_up(expr.contraction_indices, index_blocks)
    inverse_permutation = permutation ** (-1)
    index_blocks_up_permuted = [[inverse_permutation(j) for j in i if j is not None] for i in index_blocks_up]
    sorting_keys = list(enumerate(index_blocks_up_permuted))
    sorting_keys.sort(key=lambda x: x[1])
    new_perm_image_form = [i[0] for i in sorting_keys]
    new_index_blocks = [index_blocks[i] for i in new_perm_image_form]
    new_index_perm_array_form = _af_invert([j for i in new_index_blocks for j in i])
    new_args = [args[i] for i in new_perm_image_form]
    new_contraction_indices = [tuple((new_index_perm_array_form[j] for j in i)) for i in contraction_indices]
    new_expr = _array_contraction(_array_tensor_product(*new_args), *new_contraction_indices)
    new_permutation = Permutation(_af_invert([j for i in [permutation_array_blocks_up[k] for k in new_perm_image_form] for j in i]))
    return (new_expr, new_permutation)