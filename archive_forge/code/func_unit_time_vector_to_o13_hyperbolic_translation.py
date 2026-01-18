from ..matrix import vector, matrix
from ..math_basics import is_RealIntervalFieldElement
from ..sage_helper import _within_sage
from a real type (either a SnapPy.Number or one
def unit_time_vector_to_o13_hyperbolic_translation(v):
    """
    Takes a point (time-like unit vector) in the hyperboloid model and
    returns the O13-matrix corresponding to the hyperbolic translation
    moving the origin to that point (that is, the translation fixing
    the geodesic between the origin and the point and introducing no
    rotation about that geodesic).
    """
    v1 = [1 + v[0], v[1], v[2], v[3]]
    m = [[x * y / v1[0] for x in v1] for y in v1]
    m[0][0] -= 1
    m[1][1] += 1
    m[2][2] += 1
    m[3][3] += 1
    return matrix(m)