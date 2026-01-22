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
class _EditArrayContraction:
    """
    Utility class to help manipulate array contraction objects.

    This class takes as input an ``ArrayContraction`` object and turns it into
    an editable object.

    The field ``args_with_ind`` of this class is a list of ``_ArgE`` objects
    which can be used to easily edit the contraction structure of the
    expression.

    Once editing is finished, the ``ArrayContraction`` object may be recreated
    by calling the ``.to_array_contraction()`` method.
    """

    def __init__(self, base_array: typing.Union[ArrayContraction, ArrayDiagonal, ArrayTensorProduct]):
        expr: Basic
        diagonalized: tTuple[tTuple[int, ...], ...]
        contraction_indices: List[tTuple[int]]
        if isinstance(base_array, ArrayContraction):
            mapping = _get_mapping_from_subranks(base_array.subranks)
            expr = base_array.expr
            contraction_indices = base_array.contraction_indices
            diagonalized = ()
        elif isinstance(base_array, ArrayDiagonal):
            if isinstance(base_array.expr, ArrayContraction):
                mapping = _get_mapping_from_subranks(base_array.expr.subranks)
                expr = base_array.expr.expr
                diagonalized = ArrayContraction._push_indices_down(base_array.expr.contraction_indices, base_array.diagonal_indices)
                contraction_indices = base_array.expr.contraction_indices
            elif isinstance(base_array.expr, ArrayTensorProduct):
                mapping = {}
                expr = base_array.expr
                diagonalized = base_array.diagonal_indices
                contraction_indices = []
            else:
                mapping = {}
                expr = base_array.expr
                diagonalized = base_array.diagonal_indices
                contraction_indices = []
        elif isinstance(base_array, ArrayTensorProduct):
            expr = base_array
            contraction_indices = []
            diagonalized = ()
        else:
            raise NotImplementedError()
        if isinstance(expr, ArrayTensorProduct):
            args = list(expr.args)
        else:
            args = [expr]
        args_with_ind: List[_ArgE] = [_ArgE(arg) for arg in args]
        for i, contraction_tuple in enumerate(contraction_indices):
            for j in contraction_tuple:
                arg_pos, rel_pos = mapping[j]
                args_with_ind[arg_pos].indices[rel_pos] = i
        self.args_with_ind: List[_ArgE] = args_with_ind
        self.number_of_contraction_indices: int = len(contraction_indices)
        self._track_permutation: Optional[List[List[int]]] = None
        mapping = _get_mapping_from_subranks(base_array.subranks)
        for i, e in enumerate(diagonalized):
            for j in e:
                arg_pos, rel_pos = mapping[j]
                self.args_with_ind[arg_pos].indices[rel_pos] = -1 - i

    def insert_after(self, arg: _ArgE, new_arg: _ArgE):
        pos = self.args_with_ind.index(arg)
        self.args_with_ind.insert(pos + 1, new_arg)

    def get_new_contraction_index(self):
        self.number_of_contraction_indices += 1
        return self.number_of_contraction_indices - 1

    def refresh_indices(self):
        updates = {}
        for arg_with_ind in self.args_with_ind:
            updates.update({i: -1 for i in arg_with_ind.indices if i is not None})
        for i, e in enumerate(sorted(updates)):
            updates[e] = i
        self.number_of_contraction_indices = len(updates)
        for arg_with_ind in self.args_with_ind:
            arg_with_ind.indices = [updates.get(i, None) for i in arg_with_ind.indices]

    def merge_scalars(self):
        scalars = []
        for arg_with_ind in self.args_with_ind:
            if len(arg_with_ind.indices) == 0:
                scalars.append(arg_with_ind)
        for i in scalars:
            self.args_with_ind.remove(i)
        scalar = Mul.fromiter([i.element for i in scalars])
        if len(self.args_with_ind) == 0:
            self.args_with_ind.append(_ArgE(scalar))
        else:
            from sympy.tensor.array.expressions.from_array_to_matrix import _a2m_tensor_product
            self.args_with_ind[0].element = _a2m_tensor_product(scalar, self.args_with_ind[0].element)

    def to_array_contraction(self):
        counter = 0
        diag_indices = defaultdict(list)
        count_index_freq = Counter()
        for arg_with_ind in self.args_with_ind:
            count_index_freq.update(Counter(arg_with_ind.indices))
        free_index_count = count_index_freq[None]
        inv_perm1 = []
        inv_perm2 = []
        done = set()
        counter4 = 0
        for arg_with_ind in self.args_with_ind:
            counter2 = 0
            for i in arg_with_ind.indices:
                if i is None:
                    inv_perm1.append(counter4)
                    counter2 += 1
                    counter4 += 1
                    continue
                if i >= 0:
                    continue
                diag_indices[-1 - i].append(counter + counter2)
                if count_index_freq[i] == 1 and i not in done:
                    inv_perm1.append(free_index_count - 1 - i)
                    done.add(i)
                elif i not in done:
                    inv_perm2.append(free_index_count - 1 - i)
                    done.add(i)
                counter2 += 1
            arg_with_ind.indices = [i if i is not None and i >= 0 else None for i in arg_with_ind.indices]
            counter += len([i for i in arg_with_ind.indices if i is None or i < 0])
        inverse_permutation = inv_perm1 + inv_perm2
        permutation = _af_invert(inverse_permutation)
        diag_indices_filtered = [tuple(v) for v in diag_indices.values() if len(v) > 1]
        self.merge_scalars()
        self.refresh_indices()
        args = [arg.element for arg in self.args_with_ind]
        contraction_indices = self.get_contraction_indices()
        expr = _array_contraction(_array_tensor_product(*args), *contraction_indices)
        expr2 = _array_diagonal(expr, *diag_indices_filtered)
        if self._track_permutation is not None:
            permutation2 = _af_invert([j for i in self._track_permutation for j in i])
            expr2 = _permute_dims(expr2, permutation2)
        expr3 = _permute_dims(expr2, permutation)
        return expr3

    def get_contraction_indices(self) -> List[List[int]]:
        contraction_indices: List[List[int]] = [[] for i in range(self.number_of_contraction_indices)]
        current_position: int = 0
        for i, arg_with_ind in enumerate(self.args_with_ind):
            for j in arg_with_ind.indices:
                if j is not None:
                    contraction_indices[j].append(current_position)
                current_position += 1
        return contraction_indices

    def get_mapping_for_index(self, ind) -> List[_IndPos]:
        if ind >= self.number_of_contraction_indices:
            raise ValueError('index value exceeding the index range')
        positions: List[_IndPos] = []
        for i, arg_with_ind in enumerate(self.args_with_ind):
            for j, arg_ind in enumerate(arg_with_ind.indices):
                if ind == arg_ind:
                    positions.append(_IndPos(i, j))
        return positions

    def get_contraction_indices_to_ind_rel_pos(self) -> List[List[_IndPos]]:
        contraction_indices: List[List[_IndPos]] = [[] for i in range(self.number_of_contraction_indices)]
        for i, arg_with_ind in enumerate(self.args_with_ind):
            for j, ind in enumerate(arg_with_ind.indices):
                if ind is not None:
                    contraction_indices[ind].append(_IndPos(i, j))
        return contraction_indices

    def count_args_with_index(self, index: int) -> int:
        """
        Count the number of arguments that have the given index.
        """
        counter: int = 0
        for arg_with_ind in self.args_with_ind:
            if index in arg_with_ind.indices:
                counter += 1
        return counter

    def get_args_with_index(self, index: int) -> List[_ArgE]:
        """
        Get a list of arguments having the given index.
        """
        ret: List[_ArgE] = [i for i in self.args_with_ind if index in i.indices]
        return ret

    @property
    def number_of_diagonal_indices(self):
        data = set()
        for arg in self.args_with_ind:
            data.update({i for i in arg.indices if i is not None and i < 0})
        return len(data)

    def track_permutation_start(self):
        permutation = []
        perm_diag = []
        counter = 0
        counter2 = -1
        for arg_with_ind in self.args_with_ind:
            perm = []
            for i in arg_with_ind.indices:
                if i is not None:
                    if i < 0:
                        perm_diag.append(counter2)
                        counter2 -= 1
                    continue
                perm.append(counter)
                counter += 1
            permutation.append(perm)
        max_ind = max([max(i) if i else -1 for i in permutation]) if permutation else -1
        perm_diag = [max_ind - i for i in perm_diag]
        self._track_permutation = permutation + [perm_diag]

    def track_permutation_merge(self, destination: _ArgE, from_element: _ArgE):
        index_destination = self.args_with_ind.index(destination)
        index_element = self.args_with_ind.index(from_element)
        self._track_permutation[index_destination].extend(self._track_permutation[index_element])
        self._track_permutation.pop(index_element)

    def get_absolute_free_range(self, arg: _ArgE) -> typing.Tuple[int, int]:
        """
        Return the range of the free indices of the arg as absolute positions
        among all free indices.
        """
        counter = 0
        for arg_with_ind in self.args_with_ind:
            number_free_indices = len([i for i in arg_with_ind.indices if i is None])
            if arg_with_ind == arg:
                return (counter, counter + number_free_indices)
            counter += number_free_indices
        raise IndexError('argument not found')

    def get_absolute_range(self, arg: _ArgE) -> typing.Tuple[int, int]:
        """
        Return the absolute range of indices for arg, disregarding dummy
        indices.
        """
        counter = 0
        for arg_with_ind in self.args_with_ind:
            number_indices = len(arg_with_ind.indices)
            if arg_with_ind == arg:
                return (counter, counter + number_indices)
            counter += number_indices
        raise IndexError('argument not found')