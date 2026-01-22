from __future__ import annotations
import os
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from functools import partial
from io import BytesIO
from numbers import Number
from typing import (
import numpy as np
from xarray import backends, conventions
from xarray.backends import plugins
from xarray.backends.common import (
from xarray.backends.locks import _get_scheduler
from xarray.backends.zarr import open_zarr
from xarray.core import indexing
from xarray.core.combine import (
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, _get_chunk, _maybe_chunk
from xarray.core.indexes import Index
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import is_remote_uri
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import guess_chunkmanager
def _auto_detect_region(ds_new, ds_orig, dim):
    coord = ds_orig[dim]
    da_map = DataArray(np.arange(coord.size), coords={dim: coord})
    try:
        da_idxs = da_map.sel({dim: ds_new[dim]})
    except KeyError as e:
        if 'not all values found' in str(e):
            raise KeyError(f"Not all values of coordinate '{dim}' in the new array were found in the original store. Writing to a zarr region slice requires that no dimensions or metadata are changed by the write.")
        else:
            raise e
    if (da_idxs.diff(dim) != 1).any():
        raise ValueError(f"The auto-detected region of coordinate '{dim}' for writing new data to the original store had non-contiguous indices. Writing to a zarr region slice requires that the new data constitute a contiguous subset of the original store.")
    dim_slice = slice(da_idxs.values[0], da_idxs.values[-1] + 1)
    return dim_slice