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
def _nest_permutation(cls, expr, permutation):
    if isinstance(expr, ArrayTensorProduct):
        return _permute_dims(*cls._check_if_there_are_closed_cycles(expr, permutation))
    elif isinstance(expr, ArrayContraction):
        cycles = permutation.cyclic_form
        newcycles = ArrayContraction._convert_outer_indices_to_inner_indices(expr, *cycles)
        newpermutation = Permutation(newcycles)
        new_contr_indices = [tuple((newpermutation(j) for j in i)) for i in expr.contraction_indices]
        return _array_contraction(PermuteDims(expr.expr, newpermutation), *new_contr_indices)
    elif isinstance(expr, ArrayAdd):
        return _array_add(*[PermuteDims(arg, permutation) for arg in expr.args])
    return None