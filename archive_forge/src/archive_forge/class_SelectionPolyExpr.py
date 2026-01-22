import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
class SelectionPolyExpr(Selection2DExpr):

    def _skip(self, **kwargs):
        """
        Do not skip geometry selections until polygons support returning
        indexes on lasso based selections.
        """
        skip = kwargs.get('index_cols') and self._index_skip and ('geometry' not in kwargs)
        if skip:
            self._index_skip = False
        return skip

    def _get_bounds_selection(self, xdim, ydim, **kwargs):
        from .geom import Rectangles
        (x0, x1), _, (y0, y1), _ = self._get_selection(**kwargs)
        bbox = {xdim.name: (x0, x1), ydim.name: (y0, y1)}
        index_cols = kwargs.get('index_cols')
        expr = dim.pipe(spatial_bounds_select, xdim, dim(ydim), bounds=(x0, y0, x1, y1))
        if index_cols:
            selection = self[expr.apply(self, expanded=False)]
            selection_expr = self._get_index_expr(index_cols, selection)
            return (selection_expr, bbox, None)
        return (expr, bbox, Rectangles([(x0, y0, x1, y1)]))

    def _get_lasso_selection(self, xdim, ydim, geometry, **kwargs):
        from .path import Path
        bbox = {xdim.name: geometry[:, 0], ydim.name: geometry[:, 1]}
        expr = dim.pipe(spatial_poly_select, xdim, dim(ydim), geometry=geometry)
        index_cols = kwargs.get('index_cols')
        if index_cols:
            selection = self[expr.apply(self, expanded=False)]
            selection_expr = self._get_index_expr(index_cols, selection)
            return (selection_expr, bbox, None)
        return (expr, bbox, Path([np.concatenate([geometry, geometry[:1]])]))