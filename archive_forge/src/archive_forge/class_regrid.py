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
class regrid(AggregationOperation):
    """
    regrid allows resampling a HoloViews Image type using specified
    up- and downsampling functions defined using the aggregator and
    interpolation parameters respectively. By default upsampling is
    disabled to avoid unnecessarily upscaling an image that has to be
    sent to the browser. Also disables expanding the image beyond its
    original bounds avoiding unnecessarily padding the output array
    with NaN values.
    """
    aggregator = param.ClassSelector(default=rd.mean(), class_=(rd.Reduction, rd.summary, str))
    expand = param.Boolean(default=False, doc='\n       Whether the x_range and y_range should be allowed to expand\n       beyond the extent of the data.  Setting this value to True is\n       useful for the case where you want to ensure a certain size of\n       output grid, e.g. if you are doing masking or other arithmetic\n       on the grids.  A value of False ensures that the grid is only\n       just as large as it needs to be to contain the data, which will\n       be faster and use less memory if the resulting aggregate is\n       being overlaid on a much larger background.')
    interpolation = param.ObjectSelector(default='nearest', objects=['linear', 'nearest', 'bilinear', None, False], doc='\n        Interpolation method')
    upsample = param.Boolean(default=False, doc='\n        Whether to allow upsampling if the source array is smaller\n        than the requested array. Setting this value to True will\n        enable upsampling using the interpolation method, when the\n        requested width and height are larger than what is available\n        on the source grid. If upsampling is disabled (the default)\n        the width and height are clipped to what is available on the\n        source array.')

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

    def _process(self, element, key=None):
        if ds_version <= Version('0.5.0'):
            raise RuntimeError('regrid operation requires datashader>=0.6.0')
        x, y = element.kdims
        coords = tuple((element.dimension_values(d, expanded=False) for d in [x, y]))
        info = self._get_sampling(element, x, y)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        (xstart, xend), (ystart, yend) = (x_range, y_range)
        xspan, yspan = (xend - xstart, yend - ystart)
        interp = self.p.interpolation or None
        if interp == 'bilinear':
            interp = 'linear'
        if not (self.p.upsample or interp is None) and self.p.target is None:
            (x0, x1), (y0, y1) = (element.range(0), element.range(1))
            if isinstance(x0, datetime_types):
                x0, x1 = (dt_to_int(x0, 'ns'), dt_to_int(x1, 'ns'))
            if isinstance(y0, datetime_types):
                y0, y1 = (dt_to_int(y0, 'ns'), dt_to_int(y1, 'ns'))
            exspan, eyspan = (x1 - x0, y1 - y0)
            if np.isfinite(exspan) and exspan > 0 and (xspan > 0):
                width = max([min([int(xspan / exspan * len(coords[0])), width]), 1])
            else:
                width = 0
            if np.isfinite(eyspan) and eyspan > 0 and (yspan > 0):
                height = max([min([int(yspan / eyspan * len(coords[1])), height]), 1])
            else:
                height = 0
            xunit = float(xspan) / width if width else 0
            yunit = float(yspan) / height if height else 0
            xs, ys = (np.linspace(xstart + xunit / 2.0, xend - xunit / 2.0, width), np.linspace(ystart + yunit / 2.0, yend - yunit / 2.0, height))
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)
        params = dict(bounds=(x0, y0, x1, y1))
        if width == 0 or height == 0:
            if width == 0:
                params['xdensity'] = 1
            if height == 0:
                params['ydensity'] = 1
            return element.clone((xs, ys, np.zeros((height, width))), **params)
        cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)
        regridded = {}
        arrays = self._get_xarrays(element, coords, xtype, ytype)
        agg_fn = self._get_aggregator(element, self.p.aggregator, add_field=False)
        for vd, xarr in arrays.items():
            rarray = cvs.raster(xarr, upsample_method=interp, downsample_method=agg_fn)
            if xtype == 'datetime':
                rarray[x.name] = rarray[x.name].astype('datetime64[ns]')
            if ytype == 'datetime':
                rarray[y.name] = rarray[y.name].astype('datetime64[ns]')
            regridded[vd] = rarray
        regridded = xr.Dataset(regridded)
        return element.clone(regridded, datatype=['xarray'] + element.datatype, **params)