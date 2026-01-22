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
class AggregationOperation(ResampleOperation2D):
    """
    AggregationOperation extends the ResampleOperation2D defining an
    aggregator parameter used to define a datashader Reduction.
    """
    aggregator = param.ClassSelector(class_=(rd.Reduction, rd.summary, str), default=rd.count(), doc='\n        Datashader reduction function used for aggregating the data.\n        The aggregator may also define a column to aggregate; if\n        no column is defined the first value dimension of the element\n        will be used. May also be defined as a string.')
    selector = param.ClassSelector(class_=(rd.min, rd.max, rd.first, rd.last), default=None, doc='\n        Selector is a datashader reduction function used for selecting data.\n        The selector only works with aggregators which selects an item from\n        the original data. These selectors are min, max, first and last.')
    vdim_prefix = param.String(default='{kdims} ', allow_None=True, doc='\n        Prefix to prepend to value dimension name where {kdims}\n        templates in the names of the input element key dimensions.')
    _agg_methods = {'any': rd.any, 'count': rd.count, 'first': rd.first, 'last': rd.last, 'mode': rd.mode, 'mean': rd.mean, 'sum': rd.sum, 'var': rd.var, 'std': rd.std, 'min': rd.min, 'max': rd.max, 'count_cat': rd.count_cat}

    @classmethod
    def _get_aggregator(cls, element, agg, add_field=True):
        if ds15:
            agg_types = (rd.count, rd.any, rd.where)
        else:
            agg_types = (rd.count, rd.any)
        if isinstance(agg, str):
            if agg not in cls._agg_methods:
                agg_methods = sorted(cls._agg_methods)
                raise ValueError(f"Aggregation method '{agg!r}' is not known; aggregator must be one of: {agg_methods!r}")
            if agg == 'count_cat':
                agg = cls._agg_methods[agg]('__temp__')
            else:
                agg = cls._agg_methods[agg]()
        elements = element.traverse(lambda x: x, [Element])
        if add_field and getattr(agg, 'column', False) in ('__temp__', None) and (not isinstance(agg, agg_types)):
            if not elements:
                raise ValueError('Could not find any elements to apply %s operation to.' % cls.__name__)
            inner_element = elements[0]
            if isinstance(inner_element, TriMesh) and inner_element.nodes.vdims:
                field = inner_element.nodes.vdims[0].name
            elif inner_element.vdims:
                field = inner_element.vdims[0].name
            elif isinstance(element, NdOverlay):
                field = element.kdims[0].name
            else:
                raise ValueError("Could not determine dimension to apply '%s' operation to. Declare the dimension to aggregate as part of the datashader aggregator." % cls.__name__)
            agg = type(agg)(field)
        return agg

    def _empty_agg(self, element, x, y, width, height, xs, ys, agg_fn, **params):
        x = x.name if x else 'x'
        y = y.name if x else 'y'
        xarray = xr.DataArray(np.full((height, width), np.nan), dims=[y, x], coords={x: xs, y: ys})
        if width == 0:
            params['xdensity'] = 1
        if height == 0:
            params['ydensity'] = 1
        el = self.p.element_type(xarray, **params)
        if isinstance(agg_fn, ds.count_cat):
            vals = element.dimension_values(agg_fn.column, expanded=False)
            dim = element.get_dimension(agg_fn.column)
            return NdOverlay({v: el for v in vals}, dim)
        return el

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