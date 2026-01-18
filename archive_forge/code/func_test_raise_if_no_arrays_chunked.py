from __future__ import annotations
from importlib.metadata import EntryPoint
from typing import Any
import numpy as np
import pytest
from xarray.core.types import T_Chunks, T_DuckArray, T_NormalizedChunks
from xarray.namedarray._typing import _Chunks
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import (
from xarray.tests import has_dask, requires_dask
def test_raise_if_no_arrays_chunked(self, register_dummy_chunkmanager) -> None:
    with pytest.raises(TypeError, match='Expected a chunked array '):
        get_chunked_array_type(*[1.0, np.array([5, 6])])