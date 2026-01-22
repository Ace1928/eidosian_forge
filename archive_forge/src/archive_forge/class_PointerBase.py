from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
class PointerBase(ArrayBase):

    def __init__(self, child_type: TypeBase) -> None:
        super().__init__(child_type, 1)

    @staticmethod
    def _add(env, x: 'Data', y: 'Data') -> 'Data':
        from cupyx.jit import _internal_types
        if isinstance(y.ctype, Scalar) and y.ctype.dtype.kind in 'iu':
            return _internal_types.Data(f'({x.code} + {y.code})', x.ctype)
        return NotImplemented

    @staticmethod
    def _radd(env, x: 'Data', y: 'Data') -> 'Data':
        from cupyx.jit import _internal_types
        if isinstance(x.ctype, Scalar) and x.ctype.dtype.kind in 'iu':
            return _internal_types.Data(f'({x.code} + {y.code})', y.ctype)
        return NotImplemented

    @staticmethod
    def _sub(env, x: 'Data', y: 'Data') -> 'Data':
        from cupyx.jit import _internal_types
        if isinstance(y.ctype, Scalar) and y.ctype.dtype.kind in 'iu':
            return _internal_types.Data(f'({x.code} - {y.code})', x.ctype)
        if x.ctype == y.ctype:
            return _internal_types.Data(f'({x.code} - {y.code})', PtrDiff())
        return NotImplemented