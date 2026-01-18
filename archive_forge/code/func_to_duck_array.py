from __future__ import annotations
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from packaging.version import Version
from xarray.core.utils import is_scalar
from xarray.namedarray.utils import is_duck_array, is_duck_dask_array
def to_duck_array(data: Any, **kwargs: dict[str, Any]) -> duckarray[_ShapeType, _DType]:
    from xarray.core.indexing import ExplicitlyIndexed
    from xarray.namedarray.parallelcompat import get_chunked_array_type
    if is_chunked_array(data):
        chunkmanager = get_chunked_array_type(data)
        loaded_data, *_ = chunkmanager.compute(data, **kwargs)
        return loaded_data
    if isinstance(data, ExplicitlyIndexed):
        return data.get_duck_array()
    elif is_duck_array(data):
        return data
    else:
        return np.asarray(data)