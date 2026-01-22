from warnings import warn
import shapely
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep
@staticmethod
def _split_polygon_with_line(poly, splitter):
    """Split a Polygon with a LineString"""
    if not isinstance(poly, Polygon):
        raise GeometryTypeError('First argument must be a Polygon')
    if not isinstance(splitter, LineString):
        raise GeometryTypeError('Second argument must be a LineString')
    union = poly.boundary.union(splitter)
    poly = prep(poly)
    return [pg for pg in polygonize(union) if poly.contains(pg.representative_point())]