from .mcomplex_base import *
from .t3mlite import simplex
def _clean_ideal_vertices(vertices):
    """
    The SnapPea kernel gives us a large number for infinity.
    Convert it to infinity.
    """
    return [x if abs(x) < 1e+21 else Infinity for x in vertices]