import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
def _generate_distance_computation(int_type, dist_int_type):
    """
    Compute euclidean distance from current coordinate (ind_0, ind_1, ind_2) to
    the coordinates of the nearest point (z, y, x)."""
    return f'\n    {int_type} tmp = z - ind_0;\n    {dist_int_type} sq_dist = tmp * tmp;\n    tmp = y - ind_1;\n    sq_dist += tmp * tmp;\n    tmp = x - ind_2;\n    sq_dist += tmp * tmp;\n    dist[i] = sqrt(static_cast<F>(sq_dist));\n    '