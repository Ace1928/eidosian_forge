import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def _get_arrays_point(arr, include_z):
    coords = get_coordinates(arr, include_z=include_z)
    empties = is_empty(arr) | is_missing(arr)
    if empties.any():
        indices = np.nonzero(empties)[0]
        indices = indices - np.arange(len(indices))
        coords = np.insert(coords, indices, np.nan, axis=0)
    return (coords, ())