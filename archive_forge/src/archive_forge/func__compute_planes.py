from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.upper_halfspace import pgl2c_to_o13
from .hyperboloid_utilities import *
from .raytracing_data import *
def _compute_planes(self):
    c = vector(self.RF, [0.0, 0.0, 0.0, -1.0])

    def _compute_plane(tet, perm):
        m = tet.permutahedron_matrices[perm]
        v = c * pgl2c_to_o13(m)
        return vector([-v[0], v[1], v[2], v[3]])
    for tet in self.mcomplex.Tetrahedra:
        tet.R13_planes = {t3m.F0: _compute_plane(tet, (2, 3, 1, 0)), t3m.F1: _compute_plane(tet, (0, 3, 2, 1)), t3m.F2: _compute_plane(tet, (0, 1, 3, 2)), t3m.F3: _compute_plane(tet, (0, 2, 1, 3))}