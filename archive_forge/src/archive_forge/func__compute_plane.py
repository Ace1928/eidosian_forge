from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.upper_halfspace import pgl2c_to_o13
from .hyperboloid_utilities import *
from .raytracing_data import *
def _compute_plane(tet, perm):
    m = tet.permutahedron_matrices[perm]
    v = c * pgl2c_to_o13(m)
    return vector([-v[0], v[1], v[2], v[3]])