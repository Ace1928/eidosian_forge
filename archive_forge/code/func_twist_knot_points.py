import spherogram
import random
import itertools
from . import pl_utils
from .rational_linear_algebra import QQ, Matrix, Vector3
from .exceptions import GeneralPositionError
def twist_knot_points():
    """
    The knot 5_2 = K5a1
    """
    pts = [(25, 36, 5), (14, 36, -3), (14, 14, 0), (47, 14, 0), (47, 25, 0), (3, 25, 0), (3, 47, 0), (36, 47, -6), (36, 3, 4), (25, 3, -5)]
    return [[Vector3(pt) for pt in pts]]