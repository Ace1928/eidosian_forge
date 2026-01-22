import logging
import sys
import param
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from holoviews.core.data import MultiInterface
from holoviews.core.util import cartesian_product, get_param_values
from holoviews.operation import Operation
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection
from ..data import GeoPandasInterface
from ..element import (Image, Shape, Polygons, Path, Points, Contours,
from ..util import (
def _fast_process(self, element, key=None):
    from cartopy.img_transform import _determine_bounds
    proj = self.p.projection
    if proj == element.crs:
        return element
    h, w = element.interface.shape(element, gridded=True)[:2]
    xs = element.dimension_values(0)
    ys = element.dimension_values(1)
    if isinstance(element, RGB):
        rgb = element.rgb
        array = np.dstack([np.flipud(rgb.dimension_values(d, flat=False)) for d in rgb.vdims])
    else:
        array = element.dimension_values(2, flat=False)
    x0, y0, x1, y1 = element.bounds.lbrt()
    width = int(w) if self.p.width is None else self.p.width
    height = int(h) if self.p.height is None else self.p.height
    bounds = _determine_bounds(xs, ys, element.crs)
    yb = bounds['y']
    resampled = []
    xvalues = []
    for xb in bounds['x']:
        px0, py0, px1, py1 = project_extents((xb[0], yb[0], xb[1], yb[1]), element.crs, proj)
        if len(bounds['x']) > 1:
            xfraction = (xb[1] - xb[0]) / (x1 - x0)
            fraction_width = int(width * xfraction)
        else:
            fraction_width = width
        xs = np.linspace(px0, px1, fraction_width)
        ys = np.linspace(py0, py1, height)
        cxs, cys = cartesian_product([xs, ys])
        pxs, pys, _ = element.crs.transform_points(proj, np.asarray(cxs), np.asarray(cys)).T
        icxs = ((pxs - x0) / (x1 - x0) * w).astype(int)
        icys = ((pys - y0) / (y1 - y0) * h).astype(int)
        xvalues.append(xs)
        icxs[icxs < 0] = 0
        icys[icys < 0] = 0
        icxs[icxs >= w] = w - 1
        icys[icys >= h] = h - 1
        resampled_arr = array[icys, icxs]
        if isinstance(element, RGB):
            nvdims = len(element.vdims)
            resampled_arr = resampled_arr.reshape((fraction_width, height, nvdims)).transpose([1, 0, 2])
        else:
            resampled_arr = resampled_arr.reshape((fraction_width, height)).T
        resampled.append(resampled_arr)
    xs = np.concatenate(xvalues[::-1])
    resampled = np.hstack(resampled[::-1])
    datatypes = [element.interface.datatype, 'xarray', 'grid']
    data = (xs, ys)
    for i in range(len(element.vdims)):
        if resampled.ndim > 2:
            data = data + (resampled[::-1, :, i],)
        else:
            data = data + (resampled,)
    return element.clone(data, crs=proj, bounds=None, datatype=datatypes)