import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
class SelectionGeomExpr(Selection2DExpr):

    def _get_selection_dims(self):
        x0dim, y0dim, x1dim, y1dim = self.kdims
        invert_axes = self.opts.get('plot').kwargs.get('invert_axes', False)
        if invert_axes:
            x0dim, x1dim, y0dim, y1dim = (y0dim, y1dim, x0dim, x1dim)
        return (x0dim, y0dim, x1dim, y1dim)

    def _get_bounds_selection(self, x0dim, y0dim, x1dim, y1dim, **kwargs):
        from .geom import Rectangles
        (x0, x1), xcats, (y0, y1), ycats = self._get_selection(**kwargs)
        xsel = xcats or (x0, x1)
        ysel = ycats or (y0, y1)
        bbox = {x0dim.name: xsel, y0dim.name: ysel, x1dim.name: xsel, y1dim.name: ysel}
        index_cols = kwargs.get('index_cols')
        if index_cols:
            selection = self.dataset.clone(datatype=['dataframe', 'dictionary']).select(**bbox)
            selection_expr = self._get_index_expr(index_cols, selection)
            region_element = None
        else:
            x0expr = (dim(x0dim) >= x0) & (dim(x0dim) <= x1)
            y0expr = (dim(y0dim) >= y0) & (dim(y0dim) <= y1)
            x1expr = (dim(x1dim) >= x0) & (dim(x1dim) <= x1)
            y1expr = (dim(y1dim) >= y0) & (dim(y1dim) <= y1)
            selection_expr = x0expr & y0expr & x1expr & y1expr
            region_element = Rectangles([(x0, y0, x1, y1)])
        return (selection_expr, bbox, region_element)

    def _get_lasso_selection(self, x0dim, y0dim, x1dim, y1dim, geometry, **kwargs):
        from .path import Path
        bbox = {x0dim.name: geometry[:, 0], y0dim.name: geometry[:, 1], x1dim.name: geometry[:, 0], y1dim.name: geometry[:, 1]}
        expr = dim.pipe(spatial_geom_select, x0dim, dim(y0dim), dim(x1dim), dim(y1dim), geometry=geometry)
        index_cols = kwargs.get('index_cols')
        if index_cols:
            selection = self[expr.apply(self)]
            selection_expr = self._get_index_expr(index_cols, selection)
            return (selection_expr, bbox, None)
        return (expr, bbox, Path([np.concatenate([geometry, geometry[:1]])]))