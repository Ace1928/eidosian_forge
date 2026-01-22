from functools import lru_cache
from typing import Dict, Generic, Tuple, Type, TypeVar, Union, cast
import numpy as np
from scipy.sparse import (
from pennylane.data.base.attribute import AttributeInfo, DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
class DatasetSparseArray(Generic[SparseT], DatasetAttribute[HDF5Group, SparseT, SparseT]):
    """Attribute type for Scipy sparse arrays. Can accept values of any type in
    ``scipy.sparse``. Arrays are serialized using the CSR format."""
    type_id = 'sparse_array'

    def __post_init__(self, value: SparseT) -> None:
        super().__post_init__(value)
        self.info['sparse_array_class'] = type(value).__qualname__

    @property
    def sparse_array_class(self) -> Type[SparseT]:
        """Returns the class of sparse array that will be returned by the ``get_value()``
        method."""
        return cast(Type[SparseT], self._supported_sparse_dict()[self.info['sparse_array_class']])

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Union[SparseArray, SparseMatrix]], ...]:
        return (bsr_array, coo_array, csc_array, csr_array, dia_array, dok_array, lil_array, csc_matrix, csr_matrix, bsr_matrix, coo_matrix, dia_matrix, dok_matrix, lil_matrix)

    @classmethod
    def py_type(cls, value_type: Type[SparseArray]) -> str:
        """The module path of sparse array types is private, e.g ``scipy.sparse._csr.csr_array``.
        This method returns the public path e.g ``scipy.sparse.csr_array`` instead."""
        return f'scipy.sparse.{value_type.__qualname__}'

    def hdf5_to_value(self, bind: HDF5Group) -> SparseT:
        info = AttributeInfo(bind.attrs)
        value = csr_array((np.array(bind['data']), np.array(bind['indices']), np.array(bind['indptr'])), shape=tuple(bind['shape']))
        sparse_array_class = cast(Type[SparseT], self._supported_sparse_dict()[info['sparse_array_class']])
        if not isinstance(value, sparse_array_class):
            value = sparse_array_class(value)
        return value

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: SparseT) -> HDF5Group:
        if not isinstance(value, csr_array):
            csr_value = csr_array(value)
        else:
            csr_value = value
        bind = bind_parent.create_group(key)
        bind['data'] = csr_value.data
        bind['indices'] = csr_value.indices
        bind['indptr'] = csr_value.indptr
        bind['shape'] = csr_value.shape
        return bind

    @classmethod
    @lru_cache(1)
    def _supported_sparse_dict(cls) -> Dict[str, Type[Union[SparseArray, SparseMatrix]]]:
        """Returns a dict mapping sparse array class names to the class."""
        return {op.__name__: op for op in cls.consumes_types()}