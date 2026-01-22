import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def _multipoint_from_flatcoords(coords, offsets):
    points = creation.points(coords)
    multipoint_parts = np.diff(offsets)
    multipoint_indices = np.repeat(np.arange(len(multipoint_parts)), multipoint_parts)
    result = np.empty(len(offsets) - 1, dtype=object)
    result = creation.multipoints(points, indices=multipoint_indices, out=result)
    result[multipoint_parts == 0] = creation.empty(1, geom_type=GeometryType.MULTIPOINT).item()
    return result