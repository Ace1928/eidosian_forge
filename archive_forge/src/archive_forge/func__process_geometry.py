from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
def _process_geometry(geometry):
    ragged = shapely.to_ragged_array(geometry)
    geometry_type = ragged[0]
    if geometry_type not in (shapely.GeometryType.LINESTRING, shapely.GeometryType.MULTILINESTRING, shapely.GeometryType.MULTIPOLYGON, shapely.GeometryType.POLYGON):
        raise ValueError(f'Canvas.line supports GeoPandas geometry types of LINESTRING, MULTILINESTRING, MULTIPOLYGON and POLYGON, not {repr(geometry_type)}')
    coords = ragged[1].ravel()
    if geometry_type == shapely.GeometryType.LINESTRING:
        offsets = ragged[2][0]
        outer_offsets = np.arange(len(offsets))
        closed_rings = False
    elif geometry_type == shapely.GeometryType.MULTILINESTRING:
        offsets, outer_offsets = ragged[2]
        closed_rings = False
    elif geometry_type == shapely.GeometryType.MULTIPOLYGON:
        offsets, temp_offsets, outer_offsets = ragged[2]
        outer_offsets = temp_offsets[outer_offsets]
        closed_rings = True
    else:
        offsets, outer_offsets = ragged[2]
        closed_rings = True
    return (coords, offsets, outer_offsets, closed_rings)