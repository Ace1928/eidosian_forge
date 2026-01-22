from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.upper_halfspace import pgl2c_to_o13
from .hyperboloid_utilities import *
from .raytracing_data import *
def _matrices_for_tet(hyperbolic_structure, tet_num):
    RF = hyperbolic_structure.vertex_gram_matrices[0].base_ring()
    CF = RF.complex_field()
    matrices = {(0, 1, 2, 3): matrix.identity(CF, 2)}
    for new_perm, edge_type, old_perm in _new_perm_edge_type_old_perm:
        tet_edge = TruncatedComplex.Edge(edge_type, (tet_num, old_perm))
        m = hyperbolic_structure.pgl2_matrix_for_edge(tet_edge)
        matrices[new_perm] = m * matrices[old_perm.tuple()]
    return matrices