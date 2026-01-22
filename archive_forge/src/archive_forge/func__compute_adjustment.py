from ...sage_helper import _within_sage, sage_method
from .extended_bloch import *
from ...snap import t3mlite as t3m
def _compute_adjustment(mcomplex):
    """
    Given an mcomplex, compute the adjustment term to account for the
    triangulation not being ordered.

    So far, only solves for the 3-torsion but 2-torsion remains
    """
    return sum([_compute_adjustment_for_face(face) for face in mcomplex.Faces])