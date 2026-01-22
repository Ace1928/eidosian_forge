import numpy as np
from shapely.errors import GeometryTypeError
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point
from shapely.geometry.polygon import LinearRing, Polygon
def _empty_shape_for_no_coordinates(geom_type):
    """Return empty counterpart for geom_type"""
    if geom_type == 'point':
        return Point()
    elif geom_type == 'multipoint':
        return MultiPoint()
    elif geom_type == 'linestring':
        return LineString()
    elif geom_type == 'multilinestring':
        return MultiLineString()
    elif geom_type == 'polygon':
        return Polygon()
    elif geom_type == 'multipolygon':
        return MultiPolygon()
    else:
        raise GeometryTypeError(f'Unknown geometry type: {geom_type!r}')