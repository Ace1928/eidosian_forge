import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def _get_arrays_multipolygon(arr, include_z):
    arr_flat, part_indices = get_parts(arr, return_index=True)
    offsets3 = _indices_to_offsets(part_indices, len(arr))
    arr_flat2, ring_indices = get_rings(arr_flat, return_index=True)
    offsets2 = np.insert(np.bincount(ring_indices).cumsum(), 0, 0)
    coords, indices = get_coordinates(arr_flat2, return_index=True, include_z=include_z)
    offsets1 = np.insert(np.bincount(indices).cumsum(), 0, 0)
    return (coords, (offsets1, offsets2, offsets3))