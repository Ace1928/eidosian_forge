from __future__ import annotations
import copy
import warnings
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, cast, overload
import numpy as np
import pytest
from xarray.core.indexing import ExplicitlyIndexed
from xarray.namedarray._typing import (
from xarray.namedarray.core import NamedArray, from_array
def test_replace_namedarray(self) -> None:
    dtype_float = np.dtype(np.float32)
    np_val: np.ndarray[Any, np.dtype[np.float32]]
    np_val = np.array([1.5, 3.2], dtype=dtype_float)
    np_val2: np.ndarray[Any, np.dtype[np.float32]]
    np_val2 = 2 * np_val
    narr_float: NamedArray[Any, np.dtype[np.float32]]
    narr_float = NamedArray(('x',), np_val)
    assert narr_float.dtype == dtype_float
    narr_float2: NamedArray[Any, np.dtype[np.float32]]
    narr_float2 = NamedArray(('x',), np_val2)
    assert narr_float2.dtype == dtype_float

    class Variable(NamedArray[_ShapeType_co, _DType_co], Generic[_ShapeType_co, _DType_co]):

        @overload
        def _new(self, dims: _DimsLike | Default=..., data: duckarray[Any, _DType]=..., attrs: _AttrsLike | Default=...) -> Variable[Any, _DType]:
            ...

        @overload
        def _new(self, dims: _DimsLike | Default=..., data: Default=..., attrs: _AttrsLike | Default=...) -> Variable[_ShapeType_co, _DType_co]:
            ...

        def _new(self, dims: _DimsLike | Default=_default, data: duckarray[Any, _DType] | Default=_default, attrs: _AttrsLike | Default=_default) -> Variable[Any, _DType] | Variable[_ShapeType_co, _DType_co]:
            dims_ = copy.copy(self._dims) if dims is _default else dims
            attrs_: Mapping[Any, Any] | None
            if attrs is _default:
                attrs_ = None if self._attrs is None else self._attrs.copy()
            else:
                attrs_ = attrs
            if data is _default:
                return type(self)(dims_, copy.copy(self._data), attrs_)
            cls_ = cast('type[Variable[Any, _DType]]', type(self))
            return cls_(dims_, data, attrs_)
    var_float: Variable[Any, np.dtype[np.float32]]
    var_float = Variable(('x',), np_val)
    assert var_float.dtype == dtype_float
    var_float2: Variable[Any, np.dtype[np.float32]]
    var_float2 = var_float._replace(('x',), np_val2)
    assert var_float2.dtype == dtype_float