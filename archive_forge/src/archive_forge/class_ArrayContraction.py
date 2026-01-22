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
class ArrayContraction(_CodegenArrayAbstract):
    """
    This class is meant to represent contractions of arrays in a form easily
    processable by the code printers.
    """

    def __new__(cls, expr, *contraction_indices, **kwargs):
        contraction_indices = _sort_contraction_indices(contraction_indices)
        expr = _sympify(expr)
        canonicalize = kwargs.get('canonicalize', False)
        obj = Basic.__new__(cls, expr, *contraction_indices)
        obj._subranks = _get_subranks(expr)
        obj._mapping = _get_mapping_from_subranks(obj._subranks)
        free_indices_to_position = {i: i for i in range(sum(obj._subranks)) if all((i not in cind for cind in contraction_indices))}
        obj._free_indices_to_position = free_indices_to_position
        shape = get_shape(expr)
        cls._validate(expr, *contraction_indices)
        if shape:
            shape = tuple((shp for i, shp in enumerate(shape) if not any((i in j for j in contraction_indices))))
        obj._shape = shape
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        expr = self.expr
        contraction_indices = self.contraction_indices
        if len(contraction_indices) == 0:
            return expr
        if isinstance(expr, ArrayContraction):
            return self._ArrayContraction_denest_ArrayContraction(expr, *contraction_indices)
        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            return self._ArrayContraction_denest_ZeroArray(expr, *contraction_indices)
        if isinstance(expr, PermuteDims):
            return self._ArrayContraction_denest_PermuteDims(expr, *contraction_indices)
        if isinstance(expr, ArrayTensorProduct):
            expr, contraction_indices = self._sort_fully_contracted_args(expr, contraction_indices)
            expr, contraction_indices = self._lower_contraction_to_addends(expr, contraction_indices)
            if len(contraction_indices) == 0:
                return expr
        if isinstance(expr, ArrayDiagonal):
            return self._ArrayContraction_denest_ArrayDiagonal(expr, *contraction_indices)
        if isinstance(expr, ArrayAdd):
            return self._ArrayContraction_denest_ArrayAdd(expr, *contraction_indices)
        contraction_indices = [i for i in contraction_indices if len(i) > 1 or get_shape(expr)[i[0]] != 1]
        if len(contraction_indices) == 0:
            return expr
        return self.func(expr, *contraction_indices, canonicalize=False)

    def __mul__(self, other):
        if other == 1:
            return self
        else:
            raise NotImplementedError('Product of N-dim arrays is not uniquely defined. Use another method.')

    def __rmul__(self, other):
        if other == 1:
            return self
        else:
            raise NotImplementedError('Product of N-dim arrays is not uniquely defined. Use another method.')

    @staticmethod
    def _validate(expr, *contraction_indices):
        shape = get_shape(expr)
        if shape is None:
            return
        for i in contraction_indices:
            if len({shape[j] for j in i if shape[j] != -1}) != 1:
                raise ValueError('contracting indices of different dimensions')

    @classmethod
    def _push_indices_down(cls, contraction_indices, indices):
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        flattened_contraction_indices.sort()
        transform = _build_push_indices_down_func_transformation(flattened_contraction_indices)
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_up(cls, contraction_indices, indices):
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        flattened_contraction_indices.sort()
        transform = _build_push_indices_up_func_transformation(flattened_contraction_indices)
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _lower_contraction_to_addends(cls, expr, contraction_indices):
        if isinstance(expr, ArrayAdd):
            raise NotImplementedError()
        if not isinstance(expr, ArrayTensorProduct):
            return (expr, contraction_indices)
        subranks = expr.subranks
        cumranks = list(accumulate([0] + subranks))
        contraction_indices_remaining = []
        contraction_indices_args = [[] for i in expr.args]
        backshift = set()
        for i, contraction_group in enumerate(contraction_indices):
            for j in range(len(expr.args)):
                if not isinstance(expr.args[j], ArrayAdd):
                    continue
                if all((cumranks[j] <= k < cumranks[j + 1] for k in contraction_group)):
                    contraction_indices_args[j].append([k - cumranks[j] for k in contraction_group])
                    backshift.update(contraction_group)
                    break
            else:
                contraction_indices_remaining.append(contraction_group)
        if len(contraction_indices_remaining) == len(contraction_indices):
            return (expr, contraction_indices)
        total_rank = get_rank(expr)
        shifts = list(accumulate([1 if i in backshift else 0 for i in range(total_rank)]))
        contraction_indices_remaining = [Tuple.fromiter((j - shifts[j] for j in i)) for i in contraction_indices_remaining]
        ret = _array_tensor_product(*[_array_contraction(arg, *contr) for arg, contr in zip(expr.args, contraction_indices_args)])
        return (ret, contraction_indices_remaining)

    def split_multiple_contractions(self):
        """
        Recognize multiple contractions and attempt at rewriting them as paired-contractions.

        This allows some contractions involving more than two indices to be
        rewritten as multiple contractions involving two indices, thus allowing
        the expression to be rewritten as a matrix multiplication line.

        Examples:

        * `A_ij b_j0 C_jk` ===> `A*DiagMatrix(b)*C`

        Care for:
        - matrix being diagonalized (i.e. `A_ii`)
        - vectors being diagonalized (i.e. `a_i0`)

        Multiple contractions can be split into matrix multiplications if
        not more than two arguments are non-diagonals or non-vectors.
        Vectors get diagonalized while diagonal matrices remain diagonal.
        The non-diagonal matrices can be at the beginning or at the end
        of the final matrix multiplication line.
        """
        editor = _EditArrayContraction(self)
        contraction_indices = self.contraction_indices
        onearray_insert = []
        for indl, links in enumerate(contraction_indices):
            if len(links) <= 2:
                continue
            positions = editor.get_mapping_for_index(indl)
            current_dimension = self.expr.shape[links[0]]
            not_vectors = []
            vectors = []
            for arg_ind, rel_ind in positions:
                arg = editor.args_with_ind[arg_ind]
                mat = arg.element
                abs_arg_start, abs_arg_end = editor.get_absolute_range(arg)
                other_arg_pos = 1 - rel_ind
                other_arg_abs = abs_arg_start + other_arg_pos
                if 1 not in mat.shape or ((current_dimension == 1) is True and mat.shape != (1, 1)) or any((other_arg_abs in l for li, l in enumerate(contraction_indices) if li != indl)):
                    not_vectors.append((arg, rel_ind))
                else:
                    vectors.append((arg, rel_ind))
            if len(not_vectors) > 2:
                continue
            for v, rel_ind in vectors:
                v.element = diagonalize_vector(v.element)
            vectors_to_loop = not_vectors[:1] + vectors + not_vectors[1:]
            first_not_vector, rel_ind = vectors_to_loop[0]
            new_index = first_not_vector.indices[rel_ind]
            for v, rel_ind in vectors_to_loop[1:-1]:
                v.indices[rel_ind] = new_index
                new_index = editor.get_new_contraction_index()
                assert v.indices.index(None) == 1 - rel_ind
                v.indices[v.indices.index(None)] = new_index
                onearray_insert.append(v)
            last_vec, rel_ind = vectors_to_loop[-1]
            last_vec.indices[rel_ind] = new_index
        for v in onearray_insert:
            editor.insert_after(v, _ArgE(OneArray(1), [None]))
        return editor.to_array_contraction()

    def flatten_contraction_of_diagonal(self):
        if not isinstance(self.expr, ArrayDiagonal):
            return self
        contraction_down = self.expr._push_indices_down(self.expr.diagonal_indices, self.contraction_indices)
        new_contraction_indices = []
        diagonal_indices = self.expr.diagonal_indices[:]
        for i in contraction_down:
            contraction_group = list(i)
            for j in i:
                diagonal_with = [k for k in diagonal_indices if j in k]
                contraction_group.extend([l for k in diagonal_with for l in k])
                diagonal_indices = [k for k in diagonal_indices if k not in diagonal_with]
            new_contraction_indices.append(sorted(set(contraction_group)))
        new_contraction_indices = ArrayDiagonal._push_indices_up(diagonal_indices, new_contraction_indices)
        return _array_contraction(_array_diagonal(self.expr.expr, *diagonal_indices), *new_contraction_indices)

    @staticmethod
    def _get_free_indices_to_position_map(free_indices, contraction_indices):
        free_indices_to_position = {}
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        counter = 0
        for ind in free_indices:
            while counter in flattened_contraction_indices:
                counter += 1
            free_indices_to_position[ind] = counter
            counter += 1
        return free_indices_to_position

    @staticmethod
    def _get_index_shifts(expr):
        """
        Get the mapping of indices at the positions before the contraction
        occurs.

        Examples
        ========

        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> from sympy import MatrixSymbol
        >>> M = MatrixSymbol("M", 3, 3)
        >>> N = MatrixSymbol("N", 3, 3)
        >>> cg = tensorcontraction(tensorproduct(M, N), [1, 2])
        >>> cg._get_index_shifts(cg)
        [0, 2]

        Indeed, ``cg`` after the contraction has two dimensions, 0 and 1. They
        need to be shifted by 0 and 2 to get the corresponding positions before
        the contraction (that is, 0 and 3).
        """
        inner_contraction_indices = expr.contraction_indices
        all_inner = [j for i in inner_contraction_indices for j in i]
        all_inner.sort()
        total_rank = _get_subrank(expr)
        inner_rank = len(all_inner)
        outer_rank = total_rank - inner_rank
        shifts = [0 for i in range(outer_rank)]
        counter = 0
        pointer = 0
        for i in range(outer_rank):
            while pointer < inner_rank and counter >= all_inner[pointer]:
                counter += 1
                pointer += 1
            shifts[i] += pointer
            counter += 1
        return shifts

    @staticmethod
    def _convert_outer_indices_to_inner_indices(expr, *outer_contraction_indices):
        shifts = ArrayContraction._get_index_shifts(expr)
        outer_contraction_indices = tuple((tuple((shifts[j] + j for j in i)) for i in outer_contraction_indices))
        return outer_contraction_indices

    @staticmethod
    def _flatten(expr, *outer_contraction_indices):
        inner_contraction_indices = expr.contraction_indices
        outer_contraction_indices = ArrayContraction._convert_outer_indices_to_inner_indices(expr, *outer_contraction_indices)
        contraction_indices = inner_contraction_indices + outer_contraction_indices
        return _array_contraction(expr.expr, *contraction_indices)

    @classmethod
    def _ArrayContraction_denest_ArrayContraction(cls, expr, *contraction_indices):
        return cls._flatten(expr, *contraction_indices)

    @classmethod
    def _ArrayContraction_denest_ZeroArray(cls, expr, *contraction_indices):
        contraction_indices_flat = [j for i in contraction_indices for j in i]
        shape = [e for i, e in enumerate(expr.shape) if i not in contraction_indices_flat]
        return ZeroArray(*shape)

    @classmethod
    def _ArrayContraction_denest_ArrayAdd(cls, expr, *contraction_indices):
        return _array_add(*[_array_contraction(i, *contraction_indices) for i in expr.args])

    @classmethod
    def _ArrayContraction_denest_PermuteDims(cls, expr, *contraction_indices):
        permutation = expr.permutation
        plist = permutation.array_form
        new_contraction_indices = [tuple((permutation(j) for j in i)) for i in contraction_indices]
        new_plist = [i for i in plist if not any((i in j for j in new_contraction_indices))]
        new_plist = cls._push_indices_up(new_contraction_indices, new_plist)
        return _permute_dims(_array_contraction(expr.expr, *new_contraction_indices), Permutation(new_plist))

    @classmethod
    def _ArrayContraction_denest_ArrayDiagonal(cls, expr: 'ArrayDiagonal', *contraction_indices):
        diagonal_indices = list(expr.diagonal_indices)
        down_contraction_indices = expr._push_indices_down(expr.diagonal_indices, contraction_indices, get_rank(expr.expr))
        down_contraction_indices = [[k for j in i for k in (j if isinstance(j, (tuple, Tuple)) else [j])] for i in down_contraction_indices]
        new_contraction_indices = []
        for contr_indgrp in down_contraction_indices:
            ind = contr_indgrp[:]
            for j, diag_indgrp in enumerate(diagonal_indices):
                if diag_indgrp is None:
                    continue
                if any((i in diag_indgrp for i in contr_indgrp)):
                    ind.extend(diag_indgrp)
                    diagonal_indices[j] = None
            new_contraction_indices.append(sorted(set(ind)))
        new_diagonal_indices_down = [i for i in diagonal_indices if i is not None]
        new_diagonal_indices = ArrayContraction._push_indices_up(new_contraction_indices, new_diagonal_indices_down)
        return _array_diagonal(_array_contraction(expr.expr, *new_contraction_indices), *new_diagonal_indices)

    @classmethod
    def _sort_fully_contracted_args(cls, expr, contraction_indices):
        if expr.shape is None:
            return (expr, contraction_indices)
        cumul = list(accumulate([0] + expr.subranks))
        index_blocks = [list(range(cumul[i], cumul[i + 1])) for i in range(len(expr.args))]
        contraction_indices_flat = {j for i in contraction_indices for j in i}
        fully_contracted = [all((j in contraction_indices_flat for j in range(cumul[i], cumul[i + 1]))) for i, arg in enumerate(expr.args)]
        new_pos = sorted(range(len(expr.args)), key=lambda x: (0, default_sort_key(expr.args[x])) if fully_contracted[x] else (1,))
        new_args = [expr.args[i] for i in new_pos]
        new_index_blocks_flat = [j for i in new_pos for j in index_blocks[i]]
        index_permutation_array_form = _af_invert(new_index_blocks_flat)
        new_contraction_indices = [tuple((index_permutation_array_form[j] for j in i)) for i in contraction_indices]
        new_contraction_indices = _sort_contraction_indices(new_contraction_indices)
        return (_array_tensor_product(*new_args), new_contraction_indices)

    def _get_contraction_tuples(self):
        """
        Return tuples containing the argument index and position within the
        argument of the index position.

        Examples
        ========

        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)

        >>> cg = tensorcontraction(tensorproduct(A, B), (1, 2))
        >>> cg._get_contraction_tuples()
        [[(0, 1), (1, 0)]]

        Notes
        =====

        Here the contraction pair `(1, 2)` meaning that the 2nd and 3rd indices
        of the tensor product `A\\otimes B` are contracted, has been transformed
        into `(0, 1)` and `(1, 0)`, identifying the same indices in a different
        notation. `(0, 1)` is the second index (1) of the first argument (i.e.
                0 or `A`). `(1, 0)` is the first index (i.e. 0) of the second
        argument (i.e. 1 or `B`).
        """
        mapping = self._mapping
        return [[mapping[j] for j in i] for i in self.contraction_indices]

    @staticmethod
    def _contraction_tuples_to_contraction_indices(expr, contraction_tuples):
        ranks = expr.subranks
        cumulative_ranks = [0] + list(accumulate(ranks))
        return [tuple((cumulative_ranks[j] + k for j, k in i)) for i in contraction_tuples]

    @property
    def free_indices(self):
        return self._free_indices[:]

    @property
    def free_indices_to_position(self):
        return dict(self._free_indices_to_position)

    @property
    def expr(self):
        return self.args[0]

    @property
    def contraction_indices(self):
        return self.args[1:]

    def _contraction_indices_to_components(self):
        expr = self.expr
        if not isinstance(expr, ArrayTensorProduct):
            raise NotImplementedError('only for contractions of tensor products')
        ranks = expr.subranks
        mapping = {}
        counter = 0
        for i, rank in enumerate(ranks):
            for j in range(rank):
                mapping[counter] = (i, j)
                counter += 1
        return mapping

    def sort_args_by_name(self):
        """
        Sort arguments in the tensor product so that their order is lexicographical.

        Examples
        ========

        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> C = MatrixSymbol("C", N, N)
        >>> D = MatrixSymbol("D", N, N)

        >>> cg = convert_matrix_to_array(C*D*A*B)
        >>> cg
        ArrayContraction(ArrayTensorProduct(A, D, C, B), (0, 3), (1, 6), (2, 5))
        >>> cg.sort_args_by_name()
        ArrayContraction(ArrayTensorProduct(A, D, B, C), (0, 3), (1, 4), (2, 7))
        """
        expr = self.expr
        if not isinstance(expr, ArrayTensorProduct):
            return self
        args = expr.args
        sorted_data = sorted(enumerate(args), key=lambda x: default_sort_key(x[1]))
        pos_sorted, args_sorted = zip(*sorted_data)
        reordering_map = {i: pos_sorted.index(i) for i, arg in enumerate(args)}
        contraction_tuples = self._get_contraction_tuples()
        contraction_tuples = [[(reordering_map[j], k) for j, k in i] for i in contraction_tuples]
        c_tp = _array_tensor_product(*args_sorted)
        new_contr_indices = self._contraction_tuples_to_contraction_indices(c_tp, contraction_tuples)
        return _array_contraction(c_tp, *new_contr_indices)

    def _get_contraction_links(self):
        """
        Returns a dictionary of links between arguments in the tensor product
        being contracted.

        See the example for an explanation of the values.

        Examples
        ========

        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> C = MatrixSymbol("C", N, N)
        >>> D = MatrixSymbol("D", N, N)

        Matrix multiplications are pairwise contractions between neighboring
        matrices:

        `A_{ij} B_{jk} C_{kl} D_{lm}`

        >>> cg = convert_matrix_to_array(A*B*C*D)
        >>> cg
        ArrayContraction(ArrayTensorProduct(B, C, A, D), (0, 5), (1, 2), (3, 6))

        >>> cg._get_contraction_links()
        {0: {0: (2, 1), 1: (1, 0)}, 1: {0: (0, 1), 1: (3, 0)}, 2: {1: (0, 0)}, 3: {0: (1, 1)}}

        This dictionary is interpreted as follows: argument in position 0 (i.e.
        matrix `A`) has its second index (i.e. 1) contracted to `(1, 0)`, that
        is argument in position 1 (matrix `B`) on the first index slot of `B`,
        this is the contraction provided by the index `j` from `A`.

        The argument in position 1 (that is, matrix `B`) has two contractions,
        the ones provided by the indices `j` and `k`, respectively the first
        and second indices (0 and 1 in the sub-dict).  The link `(0, 1)` and
        `(2, 0)` respectively. `(0, 1)` is the index slot 1 (the 2nd) of
        argument in position 0 (that is, `A_{\\ldot j}`), and so on.
        """
        args, dlinks = _get_contraction_links([self], self.subranks, *self.contraction_indices)
        return dlinks

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, 'as_explicit'):
            expr = expr.as_explicit()
        return tensorcontraction(expr, *self.contraction_indices)