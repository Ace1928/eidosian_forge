from collections import defaultdict
from sympy import Function
from sympy.combinatorics.permutations import _af_invert
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.tensor.array.expressions import ArrayElementwiseApplyFunc
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.combinatorics import Permutation
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal, \
from sympy.tensor.array.expressions.utils import _get_argindex, _get_diagonal_indices
def _check_is_in(elem, indices):
    if elem in indices:
        return True
    if any((elem in i for i in indices if isinstance(i, frozenset))):
        return True
    return False