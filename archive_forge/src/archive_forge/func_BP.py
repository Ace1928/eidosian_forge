import itertools
from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import QQ, Vector3, Vector4, Matrix
from .barycentric_geometry import (BarycentricPoint,
from .mcomplex_with_link import McomplexWithLink
def BP(*vec):
    v = Vector4(vec)
    v = v / sum(v)
    return BarycentricPoint(*v)