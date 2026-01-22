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
class NonGeometricRaytracingData(McomplexEngine):

    def __init__(self, mcomplex):
        super(NonGeometricRaytracingData, self).__init__(mcomplex)

    def get_compile_time_constants(self):
        return {b'##num_tets##': len(self.mcomplex.Tetrahedra), b'##num_cusps##': len(self.mcomplex.Vertices)}

    def get_uniform_bindings(self):
        return {'isNonGeometric': ('bool', True), 'nonGeometricTexture': ('int', 0)}

    def initial_view_state(self):
        boost = matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        tet_num = 0
        weight = 0.0
        return (boost, tet_num, weight)

    def update_view_state(self, boost_tet_num_and_weight, m=matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])):
        boost, tet_num, weight = boost_tet_num_and_weight
        boost = boost * m
        return (boost, tet_num, weight)