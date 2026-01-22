from warnings import warn
import shapely
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep
@staticmethod
def _split_line_with_multipoint(line, splitter):
    """Split a LineString with a MultiPoint"""
    if not isinstance(line, LineString):
        raise GeometryTypeError('First argument must be a LineString')
    if not isinstance(splitter, MultiPoint):
        raise GeometryTypeError('Second argument must be a MultiPoint')
    chunks = [line]
    for pt in splitter.geoms:
        new_chunks = []
        for chunk in filter(lambda x: not x.is_empty, chunks):
            new_chunks.extend(SplitOp._split_line_with_point(chunk, pt))
        chunks = new_chunks
    return chunks