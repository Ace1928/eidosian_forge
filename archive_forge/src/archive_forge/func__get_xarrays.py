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
def _get_xarrays(self, element, coords, xtype, ytype):
    x, y = element.kdims
    dims = [y.name, x.name]
    irregular = any((element.interface.irregular(element, d) for d in dims))
    if irregular:
        coord_dict = {x.name: (('y', 'x'), coords[0]), y.name: (('y', 'x'), coords[1])}
    else:
        coord_dict = {x.name: coords[0], y.name: coords[1]}
    arrays = {}
    for i, vd in enumerate(element.vdims):
        if element.interface is XArrayInterface:
            if element.interface.packed(element):
                xarr = element.data[..., i]
            else:
                xarr = element.data[vd.name]
            if 'datetime' in (xtype, ytype):
                xarr = xarr.copy()
            if dims != xarr.dims and (not irregular):
                xarr = xarr.transpose(*dims)
        elif irregular:
            arr = element.dimension_values(vd, flat=False)
            xarr = xr.DataArray(arr, coords=coord_dict, dims=['y', 'x'])
        else:
            arr = element.dimension_values(vd, flat=False)
            xarr = xr.DataArray(arr, coords=coord_dict, dims=dims)
        if xtype == 'datetime':
            xarr[x.name] = [dt_to_int(v, 'ns') for v in xarr[x.name].values]
        if ytype == 'datetime':
            xarr[y.name] = [dt_to_int(v, 'ns') for v in xarr[y.name].values]
        arrays[vd.name] = xarr
    return arrays