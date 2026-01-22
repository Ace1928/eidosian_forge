from warnings import warn
import shapely
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep
@staticmethod
def _split_line_with_point(line, splitter):
    """Split a LineString with a Point"""
    if not isinstance(line, LineString):
        raise GeometryTypeError('First argument must be a LineString')
    if not isinstance(splitter, Point):
        raise GeometryTypeError('Second argument must be a Point')
    if not line.relate_pattern(splitter, '0********'):
        return [line]
    elif line.coords[0] == splitter.coords[0]:
        return [line]
    distance_on_line = line.project(splitter)
    coords = list(line.coords)
    current_position = 0.0
    for i in range(len(coords) - 1):
        point1 = coords[i]
        point2 = coords[i + 1]
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        segment_length = (dx ** 2 + dy ** 2) ** 0.5
        current_position += segment_length
        if distance_on_line == current_position:
            return [LineString(coords[:i + 2]), LineString(coords[i + 1:])]
        elif distance_on_line < current_position:
            return [LineString(coords[:i + 1] + [splitter.coords[0]]), LineString([splitter.coords[0]] + coords[i + 1:])]
    return [line]