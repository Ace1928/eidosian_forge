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
class overlay_aggregate(aggregate):
    """
    Optimized aggregation for NdOverlay objects by aggregating each
    Element in an NdOverlay individually avoiding having to concatenate
    items in the NdOverlay. Works by summing sum and count aggregates and
    applying appropriate masking for NaN values. Mean aggregation
    is also supported by dividing sum and count aggregates. count_cat
    aggregates are grouped by the categorical dimension and a separate
    aggregate for each category is generated.
    """

    @classmethod
    def applies(cls, element, agg_fn, line_width=None):
        return isinstance(element, NdOverlay) and (element.type is not Curve or line_width is None) and (isinstance(agg_fn, (ds.count, ds.sum, ds.mean, ds.any)) and (agg_fn.column is None or agg_fn.column not in element.kdims) or (isinstance(agg_fn, ds.count_cat) and agg_fn.column in element.kdims))

    def _process(self, element, key=None):
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        if not self.applies(element, agg_fn, line_width=self.p.line_width):
            raise ValueError('overlay_aggregate only handles aggregation of NdOverlay types with count, sum or mean reduction.')
        dims = element.last.dimensions()[0:2]
        ndims = len(dims)
        if ndims == 1:
            x, y = (dims[0], None)
        else:
            x, y = dims
        info = self._get_sampling(element, x, y, ndims)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        ((x0, x1), (y0, y1)), _ = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)
        agg_params = dict({k: v for k, v in dict(self.param.values(), **self.p).items() if k in aggregate.param}, x_range=(x0, x1), y_range=(y0, y1))
        bbox = (x0, y0, x1, y1)
        if isinstance(agg_fn, ds.count_cat):
            agg_params.update(dict(dynamic=False, aggregator=ds.count()))
            agg_fn1 = aggregate.instance(**agg_params)
            if element.ndims == 1:
                grouped = element
            else:
                grouped = element.groupby([agg_fn.column], container_type=NdOverlay, group_type=NdOverlay)
            groups = []
            for k, el in grouped.items():
                if width == 0 or height == 0:
                    agg = self._empty_agg(el, x, y, width, height, xs, ys, ds.count())
                    groups.append((k, agg))
                else:
                    agg = agg_fn1(el)
                    groups.append((k, agg.clone(agg.data, bounds=bbox)))
            return grouped.clone(groups)
        column = agg_fn.column or 'Count'
        if isinstance(agg_fn, ds.mean):
            agg_fn1 = aggregate.instance(**dict(agg_params, aggregator=ds.sum(column)))
            agg_fn2 = aggregate.instance(**dict(agg_params, aggregator=ds.count()))
        else:
            agg_fn1 = aggregate.instance(**agg_params)
            agg_fn2 = None
        is_sum = isinstance(agg_fn, ds.sum)
        is_any = isinstance(agg_fn, ds.any)
        agg, agg2, mask = (None, None, None)
        for v in element:
            new_agg = agg_fn1.process_element(v, None)
            if is_sum:
                new_mask = np.isnan(new_agg.data[column].values)
                new_agg.data = new_agg.data.fillna(0)
            if agg_fn2:
                new_agg2 = agg_fn2.process_element(v, None)
            if agg is None:
                agg = new_agg
                if is_sum:
                    mask = new_mask
                if agg_fn2:
                    agg2 = new_agg2
            else:
                if is_any:
                    agg.data |= new_agg.data
                else:
                    agg.data += new_agg.data
                if is_sum:
                    mask &= new_mask
                if agg_fn2:
                    agg2.data += new_agg2.data
        if agg2 is not None:
            agg2.data.rename({'Count': agg_fn.column}, inplace=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                agg.data /= agg2.data
        if is_sum:
            agg.data[column].values[mask] = np.nan
        return agg.clone(bounds=bbox)