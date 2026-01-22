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
def _compute_R13_horosphere_scale_for_vertex(self, tet, V0):
    vertex = tet.Class[V0]
    if not vertex.is_complete:
        return 0.0
    area = self.areas[vertex.Index]
    if area < 1e-06:
        return 0.0
    V1, V2, _ = t3m.VerticesOfFaceCounterclockwise[t3m.comp(V0)]
    cusp_length = tet.horotriangles[V0].get_real_lengths()[V0 | V1 | V2]
    scale_for_unit_length = (-2 * tet.R13_vertex_products[V1 | V2] / (tet.R13_vertex_products[V0 | V1] * tet.R13_vertex_products[V0 | V2])).sqrt()
    return scale_for_unit_length / (cusp_length * area.sqrt())