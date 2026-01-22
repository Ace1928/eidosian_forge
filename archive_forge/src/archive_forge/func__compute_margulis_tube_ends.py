from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.snap.mcomplex_base import *
from snappy.verify.cuspCrossSection import *
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
from ..upper_halfspace.ideal_point import ideal_point_to_r13
from .hyperboloid_utilities import *
from .upper_halfspace_utilities import *
from .raytracing_data import *
from math import sqrt
def _compute_margulis_tube_ends(tet, vertex):
    if tet.Class[vertex].is_complete:
        return [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]
    return [tet.cusp_to_tet_matrices[vertex] * vector([1.0, x, 0.0, 0.0]) for x in [-1.0, 1.0]]