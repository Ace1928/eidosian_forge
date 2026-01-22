import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
class Selection2DExpr(SelectionIndexExpr):
    """
    Mixin class for Cartesian 2D elements to add basic support for
    SelectionExpr streams.
    """
    _selection_dims = 2
    _selection_streams = (SelectionXY, Lasso, Selection1D)

    def _empty_region(self):
        from .geom import Rectangles
        from .path import Path
        return Rectangles([]) * Path([])

    def _get_selection(self, **kwargs):
        xcats, ycats = (None, None)
        x0, y0, x1, y1 = kwargs['bounds']
        if 'x_selection' in kwargs:
            xsel = kwargs['x_selection']
            if isinstance(xsel, list):
                xcats = xsel
                x0, x1 = (int(round(x0)), int(round(x1)))
            ysel = kwargs['y_selection']
            if isinstance(ysel, list):
                ycats = ysel
                y0, y1 = (int(round(y0)), int(round(y1)))
        if x0 > x1:
            x0, x1 = (x1, x0)
        if y0 > y1:
            y0, y1 = (y1, y0)
        return ((x0, x1), xcats, (y0, y1), ycats)

    def _get_index_expr(self, index_cols, sel):
        if len(index_cols) == 1:
            index_dim = index_cols[0]
            vals = dim(index_dim).apply(sel, expanded=False, flat=True)
            expr = dim(index_dim).isin(list(util.unique_iterator(vals)))
        else:
            get_shape = dim(self.dataset.get_dimension(), np.shape)
            index_cols = [dim(self.dataset.get_dimension(c), np.ravel) for c in index_cols]
            vals = dim(index_cols[0], util.unique_zip, *index_cols[1:]).apply(sel, expanded=True, flat=True)
            contains = dim(index_cols[0], util.lzip, *index_cols[1:]).isin(vals, object=True)
            expr = dim(contains, np.reshape, get_shape)
        return expr

    def _get_bounds_selection(self, xdim, ydim, **kwargs):
        from .geom import Rectangles
        (x0, x1), xcats, (y0, y1), ycats = self._get_selection(**kwargs)
        xsel = xcats or (x0, x1)
        ysel = ycats or (y0, y1)
        bbox = {xdim.name: xsel, ydim.name: ysel}
        index_cols = kwargs.get('index_cols')
        if index_cols:
            selection = self.dataset.clone(datatype=['dataframe', 'dictionary']).select(**bbox)
            selection_expr = self._get_index_expr(index_cols, selection)
            region_element = None
        else:
            if xcats:
                xexpr = dim(xdim).isin(xcats)
            else:
                xexpr = (dim(xdim) >= x0) & (dim(xdim) <= x1)
            if ycats:
                yexpr = dim(ydim).isin(ycats)
            else:
                yexpr = (dim(ydim) >= y0) & (dim(ydim) <= y1)
            selection_expr = xexpr & yexpr
            region_element = Rectangles([(x0, y0, x1, y1)])
        return (selection_expr, bbox, region_element)

    def _get_lasso_selection(self, xdim, ydim, geometry, **kwargs):
        from .path import Path
        bbox = {xdim.name: geometry[:, 0], ydim.name: geometry[:, 1]}
        expr = dim.pipe(spatial_select, xdim, dim(ydim), geometry=geometry)
        index_cols = kwargs.get('index_cols')
        if index_cols:
            selection = self[expr.apply(self)]
            selection_expr = self._get_index_expr(index_cols, selection)
            return (selection_expr, bbox, None)
        return (expr, bbox, Path([np.concatenate([geometry, geometry[:1]])]))

    def _get_selection_dims(self):
        from .graphs import Graph
        if isinstance(self, Graph):
            xdim, ydim = self.nodes.dimensions()[:2]
        else:
            xdim, ydim = self.dimensions()[:2]
        invert_axes = self.opts.get('plot').kwargs.get('invert_axes', False)
        if invert_axes:
            xdim, ydim = (ydim, xdim)
        return (xdim, ydim)

    def _skip(self, **kwargs):
        skip = kwargs.get('index_cols') and self._index_skip
        if skip:
            self._index_skip = False
        return skip

    def _get_selection_expr_for_stream_value(self, **kwargs):
        from .geom import Rectangles
        from .path import Path
        if kwargs.get('bounds') is None and kwargs.get('x_selection') is None and (kwargs.get('geometry') is None) and (not kwargs.get('index')):
            return (None, None, Rectangles([]) * Path([]))
        index_cols = kwargs.get('index_cols')
        dims = self._get_selection_dims()
        if kwargs.get('index') is not None and index_cols is not None:
            expr, _, _ = self._get_index_selection(kwargs['index'], index_cols)
            return (expr, None, self._empty_region())
        elif self._skip(**kwargs):
            return None
        elif 'bounds' in kwargs:
            expr, bbox, region = self._get_bounds_selection(*dims, **kwargs)
            return (expr, bbox, None if region is None else region * Path([]))
        elif 'geometry' in kwargs:
            expr, bbox, region = self._get_lasso_selection(*dims, **kwargs)
            return (expr, bbox, None if region is None else Rectangles([]) * region)

    @staticmethod
    def _merge_regions(region1, region2, operation):
        if region1 is None or operation == 'overwrite':
            return region2
        rect1 = region1.get(0)
        rect2 = region2.get(0)
        rects = rect1.clone(rect1.interface.concatenate([rect1, rect2]))
        poly1 = region1.get(1)
        poly2 = region2.get(1)
        polys = poly1.clone([poly1, poly2])
        return rects * polys