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
def _cusp_view_matrix(tet, subsimplex, area):
    m_translation, l_translation = tet.Class[subsimplex].Translations
    CF = m_translation.parent()
    RF = m_translation.real().parent()
    translation = (m_translation + l_translation) / 2
    factor_to_move_inside = 1.0001
    rotation = l_translation / abs(l_translation)
    scale = factor_to_move_inside / area.sqrt()
    borel_transform = matrix([[scale * rotation, translation], [0, 1]], ring=CF)
    base_camera_matrix = matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], ring=RF)
    o13_matrix = tet.cusp_to_tet_matrices[subsimplex] * pgl2c_to_o13(borel_transform) * base_camera_matrix
    return o13_matrix