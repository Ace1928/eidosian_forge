import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
def to_polygons(self, transform=None, width=0, height=0, closed_only=True):
    """
        Convert this path to a list of polygons or polylines.  Each
        polygon/polyline is an (N, 2) array of vertices.  In other words,
        each polygon has no `MOVETO` instructions or curves.  This
        is useful for displaying in backends that do not support
        compound paths or Bézier curves.

        If *width* and *height* are both non-zero then the lines will
        be simplified so that vertices outside of (0, 0), (width,
        height) will be clipped.

        If *closed_only* is `True` (default), only closed polygons,
        with the last point being the same as the first point, will be
        returned.  Any unclosed polylines in the path will be
        explicitly closed.  If *closed_only* is `False`, any unclosed
        polygons in the path will be returned as unclosed polygons,
        and the closed polygons will be returned explicitly closed by
        setting the last point to the same as the first point.
        """
    if len(self.vertices) == 0:
        return []
    if transform is not None:
        transform = transform.frozen()
    if self.codes is None and (width == 0 or height == 0):
        vertices = self.vertices
        if closed_only:
            if len(vertices) < 3:
                return []
            elif np.any(vertices[0] != vertices[-1]):
                vertices = [*vertices, vertices[0]]
        if transform is None:
            return [vertices]
        else:
            return [transform.transform(vertices)]
    return _path.convert_path_to_polygons(self, transform, width, height, closed_only)