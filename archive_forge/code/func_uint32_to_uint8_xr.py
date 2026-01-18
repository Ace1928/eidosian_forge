import warnings
from collections.abc import Callable, Iterable
from functools import partial
import dask.dataframe as dd
import datashader as ds
import datashader.reductions as rd
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import param
import xarray as xr
from datashader.colors import color_lookup
from packaging.version import Version
from param.parameterized import bothmethod
from ..core import (
from ..core.data import (
from ..core.util import (
from ..element import (
from ..element.util import connect_tri_edges_pd
from ..streams import PointerXY
from .resample import LinkableOperation, ResampleOperation2D
@classmethod
def uint32_to_uint8_xr(cls, img):
    """
        Cast uint32 xarray DataArray to 4 uint8 channels.
        """
    new_array = img.values.view(dtype=np.uint8).reshape(img.shape + (4,))
    coords = dict(list(img.coords.items()) + [('band', [0, 1, 2, 3])])
    return xr.DataArray(new_array, coords=coords, dims=img.dims + ('band',))