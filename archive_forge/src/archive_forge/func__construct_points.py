import numpy as np
import shapely
from shapely.prepared import PreparedGeometry
def _construct_points(x, y):
    x, y = (np.asanyarray(x), np.asanyarray(y))
    if x.shape != y.shape:
        raise ValueError('X and Y shapes must be equivalent.')
    if x.dtype != np.float64:
        x = x.astype(np.float64)
    if y.dtype != np.float64:
        y = y.astype(np.float64)
    return shapely.points(x, y)