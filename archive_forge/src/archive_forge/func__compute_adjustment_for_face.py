from ...sage_helper import _within_sage, sage_method
from .extended_bloch import *
from ...snap import t3mlite as t3m
def _compute_adjustment_for_face(face):
    canonical_corner = face.Corners[0]
    tet = canonical_corner.Tetrahedron
    F = canonical_corner.Subsimplex
    gluing = tet.Gluing[F]
    other_F = gluing.image(F)
    return -2 * _perm_for_q_tet(F, gluing)[0] * (-1) ** t3m.FaceIndex[other_F]