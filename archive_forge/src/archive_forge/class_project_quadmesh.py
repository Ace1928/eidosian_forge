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
class project_quadmesh(_project_operation):
    supported_types = [QuadMesh]

    def _process_element(self, element):
        proj = self.p.projection
        irregular = any((element.interface.irregular(element, kd) for kd in element.kdims))
        zs = element.dimension_values(2, flat=False)
        if irregular:
            X, Y = (np.asarray(element.interface.coords(element, kd, expanded=True, edges=False)) for kd in element.kdims)
        else:
            X = element.interface.coords(element, 0, True, True, False)
            if np.all(X[0, 1:] < X[0, :-1]):
                X = X[:, ::-1]
            Y = element.interface.coords(element, 1, True, True, False)
            if np.all(Y[1:, 0] < Y[:-1, 0]):
                Y = Y[::-1, :]
        if X.shape != zs.shape:
            X = X[:-1] + np.diff(X, axis=0) / 2.0
            X = X[:, :-1] + np.diff(X, axis=1) / 2.0
        if Y.shape != zs.shape:
            Y = Y[:-1] + np.diff(Y, axis=0) / 2.0
            Y = Y[:, :-1] + np.diff(Y, axis=1) / 2.0
        coords = proj.transform_points(element.crs, X, Y)
        PX, PY = (coords[..., 0], coords[..., 1])
        wrap_proj_types = (ccrs._RectangularProjection, ccrs._WarpedRectangularProjection, ccrs.InterruptedGoodeHomolosine, ccrs.Mercator)
        if isinstance(proj, wrap_proj_types):
            with np.errstate(invalid='ignore'):
                edge_lengths = np.hypot(np.diff(PX, axis=1), np.diff(PY, axis=1))
                to_mask = (edge_lengths >= abs(proj.x_limits[1] - proj.x_limits[0]) / 2) | np.isnan(edge_lengths)
            if np.any(to_mask):
                mask = np.zeros(zs.shape, dtype=np.bool_)
                mask[:, 1:][to_mask] = True
                mask[:, 2:][to_mask[:, :-1]] = True
                mask[:, :-1][to_mask] = True
                mask[:, :-2][to_mask[:, 1:]] = True
                mask[1:, 1:][to_mask[:-1]] = True
                mask[1:, :-1][to_mask[:-1]] = True
                mask[:-1, 1:][to_mask[1:]] = True
                mask[:-1, :-1][to_mask[1:]] = True
                zs[mask] = np.nan
        params = get_param_values(element)
        return element.clone((PX, PY, zs), crs=self.p.projection, **params)