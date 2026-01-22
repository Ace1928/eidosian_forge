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
def _compute_cusp_to_tet_and_inverse_matrices(tet, vertex, i):
    trig = tet.horotriangles[vertex]
    otherVerts = [t3m.ZeroSubsimplices[(i + j) % 4] for j in range(1, 4)]
    tet_vertices = [tet.complex_vertices[v] for v in otherVerts]
    cusp_vertices = [trig.vertex_positions[vertex | v] for v in otherVerts]
    if not tet.Class[vertex].is_complete:
        z0 = cusp_vertices[0]
        cusp_vertices = [z / z0 for z in cusp_vertices]
    std_to_tet = pgl2_matrix_taking_0_1_inf_to_given_points(*tet_vertices)
    cusp_to_std = sl2c_inverse(pgl2_matrix_taking_0_1_inf_to_given_points(*cusp_vertices))
    return (pgl2c_to_o13(std_to_tet * cusp_to_std), pgl2c_to_o13(sl2c_inverse(std_to_tet * cusp_to_std)))