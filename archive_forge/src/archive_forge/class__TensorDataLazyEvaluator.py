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
class _TensorDataLazyEvaluator(CantSympify):
    """
    EXPERIMENTAL: do not rely on this class, it may change without deprecation
    warnings in future versions of SymPy.

    Explanation
    ===========

    This object contains the logic to associate components data to a tensor
    expression. Components data are set via the ``.data`` property of tensor
    expressions, is stored inside this class as a mapping between the tensor
    expression and the ``ndarray``.

    Computations are executed lazily: whereas the tensor expressions can have
    contractions, tensor products, and additions, components data are not
    computed until they are accessed by reading the ``.data`` property
    associated to the tensor expression.
    """
    _substitutions_dict: dict[Any, Any] = {}
    _substitutions_dict_tensmul: dict[Any, Any] = {}

    def __getitem__(self, key):
        dat = self._get(key)
        if dat is None:
            return None
        from .array import NDimArray
        if not isinstance(dat, NDimArray):
            return dat
        if dat.rank() == 0:
            return dat[()]
        elif dat.rank() == 1 and len(dat) == 1:
            return dat[0]
        return dat

    def _get(self, key):
        """
        Retrieve ``data`` associated with ``key``.

        Explanation
        ===========

        This algorithm looks into ``self._substitutions_dict`` for all
        ``TensorHead`` in the ``TensExpr`` (or just ``TensorHead`` if key is a
        TensorHead instance). It reconstructs the components data that the
        tensor expression should have by performing on components data the
        operations that correspond to the abstract tensor operations applied.

        Metric tensor is handled in a different manner: it is pre-computed in
        ``self._substitutions_dict_tensmul``.
        """
        if key in self._substitutions_dict:
            return self._substitutions_dict[key]
        if isinstance(key, TensorHead):
            return None
        if isinstance(key, Tensor):
            signature = tuple([i.is_up for i in key.get_indices()])
            srch = (key.component,) + signature
            if srch in self._substitutions_dict_tensmul:
                return self._substitutions_dict_tensmul[srch]
            array_list = [self.data_from_tensor(key)]
            return self.data_contract_dum(array_list, key.dum, key.ext_rank)
        if isinstance(key, TensMul):
            tensmul_args = key.args
            if len(tensmul_args) == 1 and len(tensmul_args[0].components) == 1:
                signature = tuple([i.is_up for i in tensmul_args[0].get_indices()])
                srch = (tensmul_args[0].components[0],) + signature
                if srch in self._substitutions_dict_tensmul:
                    return self._substitutions_dict_tensmul[srch]
            data_list = [self.data_from_tensor(i) if isinstance(i, Tensor) else i.data for i in tensmul_args if isinstance(i, TensExpr)]
            coeff = prod([i for i in tensmul_args if not isinstance(i, TensExpr)])
            if all((i is None for i in data_list)):
                return None
            if any((i is None for i in data_list)):
                raise ValueError('Mixing tensors with associated components data with tensors without components data')
            data_result = self.data_contract_dum(data_list, key.dum, key.ext_rank)
            return coeff * data_result
        if isinstance(key, TensAdd):
            data_list = []
            free_args_list = []
            for arg in key.args:
                if isinstance(arg, TensExpr):
                    data_list.append(arg.data)
                    free_args_list.append([x[0] for x in arg.free])
                else:
                    data_list.append(arg)
                    free_args_list.append([])
            if all((i is None for i in data_list)):
                return None
            if any((i is None for i in data_list)):
                raise ValueError('Mixing tensors with associated components data with tensors without components data')
            sum_list = []
            from .array import permutedims
            for data, free_args in zip(data_list, free_args_list):
                if len(free_args) < 2:
                    sum_list.append(data)
                else:
                    free_args_pos = {y: x for x, y in enumerate(free_args)}
                    axes = [free_args_pos[arg] for arg in key.free_args]
                    sum_list.append(permutedims(data, axes))
            return reduce(lambda x, y: x + y, sum_list)
        return None

    @staticmethod
    def data_contract_dum(ndarray_list, dum, ext_rank):
        from .array import tensorproduct, tensorcontraction, MutableDenseNDimArray
        arrays = list(map(MutableDenseNDimArray, ndarray_list))
        prodarr = tensorproduct(*arrays)
        return tensorcontraction(prodarr, *dum)

    def data_tensorhead_from_tensmul(self, data, tensmul, tensorhead):
        """
        This method is used when assigning components data to a ``TensMul``
        object, it converts components data to a fully contravariant ndarray,
        which is then stored according to the ``TensorHead`` key.
        """
        if data is None:
            return None
        return self._correct_signature_from_indices(data, tensmul.get_indices(), tensmul.free, tensmul.dum, True)

    def data_from_tensor(self, tensor):
        """
        This method corrects the components data to the right signature
        (covariant/contravariant) using the metric associated with each
        ``TensorIndexType``.
        """
        tensorhead = tensor.component
        if tensorhead.data is None:
            return None
        return self._correct_signature_from_indices(tensorhead.data, tensor.get_indices(), tensor.free, tensor.dum)

    def _assign_data_to_tensor_expr(self, key, data):
        if isinstance(key, TensAdd):
            raise ValueError('cannot assign data to TensAdd')
        if len(key.components) != 1:
            raise ValueError('cannot assign data to TensMul with multiple components')
        tensorhead = key.components[0]
        newdata = self.data_tensorhead_from_tensmul(data, key, tensorhead)
        return (tensorhead, newdata)

    def _check_permutations_on_data(self, tens, data):
        from .array import permutedims
        from .array.arrayop import Flatten
        if isinstance(tens, TensorHead):
            rank = tens.rank
            generators = tens.symmetry.generators
        elif isinstance(tens, Tensor):
            rank = tens.rank
            generators = tens.components[0].symmetry.generators
        elif isinstance(tens, TensorIndexType):
            rank = tens.metric.rank
            generators = tens.metric.symmetry.generators
        for gener in generators:
            sign_change = +1 if gener(rank) == rank else -1
            data_swapped = data
            last_data = data
            permute_axes = list(map(gener, range(rank)))
            for i in range(gener.order() - 1):
                data_swapped = permutedims(data_swapped, permute_axes)
                if any(Flatten(last_data - sign_change * data_swapped)):
                    raise ValueError('Component data symmetry structure error')
                last_data = data_swapped

    def __setitem__(self, key, value):
        """
        Set the components data of a tensor object/expression.

        Explanation
        ===========

        Components data are transformed to the all-contravariant form and stored
        with the corresponding ``TensorHead`` object. If a ``TensorHead`` object
        cannot be uniquely identified, it will raise an error.
        """
        data = _TensorDataLazyEvaluator.parse_data(value)
        self._check_permutations_on_data(key, data)
        if not isinstance(key, (TensorHead, TensorIndexType)):
            key, data = self._assign_data_to_tensor_expr(key, data)
        if isinstance(key, TensorHead):
            for dim, indextype in zip(data.shape, key.index_types):
                if indextype.data is None:
                    raise ValueError('index type {} has no components data associated (needed to raise/lower index)'.format(indextype))
                if not indextype.dim.is_number:
                    continue
                if dim != indextype.dim:
                    raise ValueError('wrong dimension of ndarray')
        self._substitutions_dict[key] = data

    def __delitem__(self, key):
        del self._substitutions_dict[key]

    def __contains__(self, key):
        return key in self._substitutions_dict

    def add_metric_data(self, metric, data):
        """
        Assign data to the ``metric`` tensor. The metric tensor behaves in an
        anomalous way when raising and lowering indices.

        Explanation
        ===========

        A fully covariant metric is the inverse transpose of the fully
        contravariant metric (it is meant matrix inverse). If the metric is
        symmetric, the transpose is not necessary and mixed
        covariant/contravariant metrics are Kronecker deltas.
        """
        self._substitutions_dict_tensmul[metric, True, True] = data
        inverse_transpose = self.inverse_transpose_matrix(data)
        self._substitutions_dict_tensmul[metric, False, False] = inverse_transpose
        m = data.tomatrix()
        invt = inverse_transpose.tomatrix()
        self._substitutions_dict_tensmul[metric, True, False] = m * invt
        self._substitutions_dict_tensmul[metric, False, True] = invt * m

    @staticmethod
    def _flip_index_by_metric(data, metric, pos):
        from .array import tensorproduct, tensorcontraction
        mdim = metric.rank()
        ddim = data.rank()
        if pos == 0:
            data = tensorcontraction(tensorproduct(metric, data), (1, mdim + pos))
        else:
            data = tensorcontraction(tensorproduct(data, metric), (pos, ddim))
        return data

    @staticmethod
    def inverse_matrix(ndarray):
        m = ndarray.tomatrix().inv()
        return _TensorDataLazyEvaluator.parse_data(m)

    @staticmethod
    def inverse_transpose_matrix(ndarray):
        m = ndarray.tomatrix().inv().T
        return _TensorDataLazyEvaluator.parse_data(m)

    @staticmethod
    def _correct_signature_from_indices(data, indices, free, dum, inverse=False):
        """
        Utility function to correct the values inside the components data
        ndarray according to whether indices are covariant or contravariant.

        It uses the metric matrix to lower values of covariant indices.
        """
        for i, indx in enumerate(indices):
            if not indx.is_up and (not inverse):
                data = _TensorDataLazyEvaluator._flip_index_by_metric(data, indx.tensor_index_type.data, i)
            elif not indx.is_up and inverse:
                data = _TensorDataLazyEvaluator._flip_index_by_metric(data, _TensorDataLazyEvaluator.inverse_matrix(indx.tensor_index_type.data), i)
        return data

    @staticmethod
    def _sort_data_axes(old, new):
        from .array import permutedims
        new_data = old.data.copy()
        old_free = [i[0] for i in old.free]
        new_free = [i[0] for i in new.free]
        for i in range(len(new_free)):
            for j in range(i, len(old_free)):
                if old_free[j] == new_free[i]:
                    old_free[i], old_free[j] = (old_free[j], old_free[i])
                    new_data = permutedims(new_data, (i, j))
                    break
        return new_data

    @staticmethod
    def add_rearrange_tensmul_parts(new_tensmul, old_tensmul):

        def sorted_compo():
            return _TensorDataLazyEvaluator._sort_data_axes(old_tensmul, new_tensmul)
        _TensorDataLazyEvaluator._substitutions_dict[new_tensmul] = sorted_compo()

    @staticmethod
    def parse_data(data):
        """
        Transform ``data`` to array. The parameter ``data`` may
        contain data in various formats, e.g. nested lists, SymPy ``Matrix``,
        and so on.

        Examples
        ========

        >>> from sympy.tensor.tensor import _TensorDataLazyEvaluator
        >>> _TensorDataLazyEvaluator.parse_data([1, 3, -6, 12])
        [1, 3, -6, 12]

        >>> _TensorDataLazyEvaluator.parse_data([[1, 2], [4, 7]])
        [[1, 2], [4, 7]]
        """
        from .array import MutableDenseNDimArray
        if not isinstance(data, MutableDenseNDimArray):
            if len(data) == 2 and hasattr(data[0], '__call__'):
                data = MutableDenseNDimArray(data[0], data[1])
            else:
                data = MutableDenseNDimArray(data)
        return data