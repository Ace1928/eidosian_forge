import numpy as np
from shapely import creation
from shapely._geometry import (
from shapely.coordinates import get_coordinates
from shapely.predicates import is_empty, is_missing
def _indices_to_offsets(indices, n):
    offsets = np.insert(np.bincount(indices).cumsum(), 0, 0)
    if len(offsets) != n + 1:
        offsets = np.pad(offsets, (0, n + 1 - len(offsets)), 'constant', constant_values=offsets[-1])
    return offsets