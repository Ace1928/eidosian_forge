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
class geom_aggregate(AggregationOperation):
    """
    Baseclass for aggregation of Geom elements.
    """
    __abstract = True

    def _aggregate(self, cvs, df, x0, y0, x1, y1, agg):
        raise NotImplementedError

    def _process(self, element, key=None):
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        x0d, y0d, x1d, y1d = element.kdims
        info = self._get_sampling(element, [x0d, x1d], [y0d, y1d], ndim=1)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)
        df = element.interface.as_dframe(element)
        if xtype == 'datetime' or ytype == 'datetime':
            df = df.copy()
        if xtype == 'datetime':
            df[x0d.name] = cast_array_to_int64(df[x0d.name].astype('datetime64[ns]'))
            df[x1d.name] = cast_array_to_int64(df[x1d.name].astype('datetime64[ns]'))
        if ytype == 'datetime':
            df[y0d.name] = cast_array_to_int64(df[y0d.name].astype('datetime64[ns]'))
            df[y1d.name] = cast_array_to_int64(df[y1d.name].astype('datetime64[ns]'))
        if isinstance(agg_fn, ds.count_cat) and df[agg_fn.column].dtype.name != 'category':
            df[agg_fn.column] = df[agg_fn.column].astype('category')
        params = self._get_agg_params(element, x0d, y0d, agg_fn, (x0, y0, x1, y1))
        if width == 0 or height == 0:
            return self._empty_agg(element, x0d, y0d, width, height, xs, ys, agg_fn, **params)
        cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)
        agg = self._aggregate(cvs, df, x0d.name, y0d.name, x1d.name, y1d.name, agg_fn)
        xdim, ydim = list(agg.dims)[:2][::-1]
        if xtype == 'datetime':
            agg[xdim] = agg[xdim].astype('datetime64[ns]')
        if ytype == 'datetime':
            agg[ydim] = agg[ydim].astype('datetime64[ns]')
        params['kdims'] = [xdim, ydim]
        if agg.ndim == 2:
            eldata = agg if ds_version > Version('0.5.0') else (xs, ys, agg.data)
            return self.p.element_type(eldata, **params)
        else:
            layers = {}
            for c in agg.coords[agg_fn.column].data:
                cagg = agg.sel(**{agg_fn.column: c})
                eldata = cagg if ds_version > Version('0.5.0') else (xs, ys, cagg.data)
                layers[c] = self.p.element_type(eldata, **params)
            return NdOverlay(layers, kdims=[element.get_dimension(agg_fn.column)])