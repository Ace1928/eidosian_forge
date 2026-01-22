from .simplex import *
from .tetrahedron import Tetrahedron
from .corner import Corner
from .arrow import Arrow
from .face import Face
from .edge import Edge
from .vertex import Vertex
from .surface import Surface, SpunSurface, ClosedSurface, ClosedSurfaceInCusped
from .perm4 import Perm4, inv
from . import files
from . import linalg
from . import homology
import sys
import random
import io
def _face_permits_two_to_three(self, a, b):
    S, T = (a.Tetrahedron, b.Tetrahedron)
    if S is None:
        return (False, 'Tetrahedron not attached to face')
    if S == T:
        return (False, 'Two tetrahedra are the same')
    return (True, None)