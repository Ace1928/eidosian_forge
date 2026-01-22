import spherogram
import random
import itertools
from . import pl_utils
from .rational_linear_algebra import QQ, Matrix, Vector3
from .exceptions import GeneralPositionError
def arcs_from_points(points_list):
    arcs = []
    for p in points_list:
        n = len(p)
        arcs += [(p[k], p[(k + 1) % n]) for k in range(n)]
    return arcs