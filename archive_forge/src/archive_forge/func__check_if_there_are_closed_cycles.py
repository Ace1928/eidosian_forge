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
def _check_if_there_are_closed_cycles(cls, expr, permutation):
    args = list(expr.args)
    subranks = expr.subranks
    cyclic_form = permutation.cyclic_form
    cumulative_subranks = [0] + list(accumulate(subranks))
    cyclic_min = [min(i) for i in cyclic_form]
    cyclic_max = [max(i) for i in cyclic_form]
    cyclic_keep = []
    for i, cycle in enumerate(cyclic_form):
        flag = True
        for j in range(len(cumulative_subranks) - 1):
            if cyclic_min[i] >= cumulative_subranks[j] and cyclic_max[i] < cumulative_subranks[j + 1]:
                args[j] = _permute_dims(args[j], Permutation([[k - cumulative_subranks[j] for k in cyclic_form[i]]]))
                flag = False
                break
        if flag:
            cyclic_keep.append(cyclic_form[i])
    return (_array_tensor_product(*args), Permutation(cyclic_keep, size=permutation.size))