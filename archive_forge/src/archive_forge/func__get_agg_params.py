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
def _get_agg_params(self, element, x, y, agg_fn, bounds):
    params = dict(get_param_values(element), kdims=[x, y], datatype=['xarray'], bounds=bounds)
    if self.vdim_prefix:
        kdim_list = '_'.join((str(kd) for kd in params['kdims']))
        vdim_prefix = self.vdim_prefix.format(kdims=kdim_list)
    else:
        vdim_prefix = ''
    category = None
    if hasattr(agg_fn, 'reduction'):
        category = agg_fn.cat_column
        agg_fn = agg_fn.reduction
    if isinstance(agg_fn, rd.summary):
        column = None
    else:
        column = agg_fn.column if agg_fn else None
    agg_name = type(agg_fn).__name__.title()
    if agg_name == 'Where':
        col = agg_fn.column if not isinstance(agg_fn.column, rd.SpecialColumn) else agg_fn.selector.column
        vdims = sorted(params['vdims'], key=lambda x: x != col)
    elif agg_name == 'Summary':
        vdims = list(agg_fn.keys)
    elif column:
        dims = [d for d in element.dimensions('ranges') if d == column]
        if not dims:
            raise ValueError("Aggregation column '{}' not found on '{}' element. Ensure the aggregator references an existing dimension.".format(column, element))
        if isinstance(agg_fn, (ds.count, ds.count_cat)):
            if vdim_prefix:
                vdim_name = f'{vdim_prefix}{column} Count'
            else:
                vdim_name = f'{column} Count'
            vdims = dims[0].clone(vdim_name, nodata=0)
        else:
            vdims = dims[0].clone(vdim_prefix + column)
    elif category:
        agg_label = f'{category} {agg_name}'
        vdims = Dimension(f'{vdim_prefix}{agg_label}', label=agg_label)
        if agg_name in ('Count', 'Any'):
            vdims.nodata = 0
    else:
        vdims = Dimension(f'{vdim_prefix}{agg_name}', label=agg_name, nodata=0)
    params['vdims'] = vdims
    return params