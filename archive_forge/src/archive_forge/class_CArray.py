from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class CArray(ArrayBase):
    from cupyx.jit import _internal_types

    def __init__(self, dtype: npt.DTypeLike, ndim: int, is_c_contiguous: bool, index_32_bits: bool) -> None:
        self.dtype = numpy.dtype(dtype)
        self._ndim = ndim
        self._c_contiguous = is_c_contiguous
        self._index_32_bits = index_32_bits
        super().__init__(Scalar(dtype), ndim)

    @classmethod
    def from_ndarray(cls, x: cupy.ndarray) -> 'CArray':
        return CArray(x.dtype, x.ndim, x._c_contiguous, x._index_32_bits)

    def size(self, instance: 'Data') -> 'Data':
        from cupyx.jit import _internal_types
        return _internal_types.Data(f'static_cast<long long>({instance.code}.size())', Scalar('q'))

    def shape(self, instance: 'Data') -> 'Data':
        from cupyx.jit import _internal_types
        if self._ndim > 10:
            raise NotImplementedError('getting shape/strides for an array with ndim > 10 is not supported yet')
        return _internal_types.Data(f'{instance.code}.get_shape()', Tuple([PtrDiff()] * self._ndim))

    def strides(self, instance: 'Data') -> 'Data':
        from cupyx.jit import _internal_types
        if self._ndim > 10:
            raise NotImplementedError('getting shape/strides for an array with ndim > 10 is not supported yet')
        return _internal_types.Data(f'{instance.code}.get_strides()', Tuple([PtrDiff()] * self._ndim))

    @_internal_types.wraps_class_method
    def begin(self, env, instance: 'Data', *args) -> 'Data':
        from cupyx.jit import _internal_types
        if self._ndim != 1:
            raise NotImplementedError('getting begin iterator for an array with ndim != 1 is not supported yet')
        method_name = 'begin_ptr' if self._c_contiguous else 'begin'
        return _internal_types.Data(f'{instance.code}.{method_name}()', CArrayIterator(instance.ctype))

    @_internal_types.wraps_class_method
    def end(self, env, instance: 'Data', *args) -> 'Data':
        from cupyx.jit import _internal_types
        if self._ndim != 1:
            raise NotImplementedError('getting end iterator for an array with ndim != 1 is not supported yet')
        method_name = 'end_ptr' if self._c_contiguous else 'end'
        return _internal_types.Data(f'{instance.code}.{method_name}()', CArrayIterator(instance.ctype))

    def __str__(self) -> str:
        ctype = get_typename(self.dtype)
        ndim = self._ndim
        c_contiguous = get_cuda_code_from_constant(self._c_contiguous, bool_)
        index_32_bits = get_cuda_code_from_constant(self._index_32_bits, bool_)
        return f'CArray<{ctype}, {ndim}, {c_contiguous}, {index_32_bits}>'

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))