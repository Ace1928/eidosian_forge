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
def zero_to_two(self, arrow1, gap):
    """
        Blow up two adjacent faces into a pair of tetrahedra.  The
        faces are specified by passing an arrow specifying the first
        face and an integer n.  The second face is obtained by
        reversing the arrow and applying next() n times.  Thus there
        are n faces between the two that are involved in the blow up.
        Returns ``True`` on success, ``False`` if the move cannot be
        performed.
        """
    arrow2 = arrow1.copy().reverse()
    count = 0
    while count < gap:
        if arrow2.next() is None:
            return False
        count = count + 1
    if arrow1.face_class() == arrow2.face_class():
        return 0
    a = arrow1.glued()
    b = arrow2.glued()
    c = self.new_arrows(2)
    c[0].glue(c[1])
    c[1].glue(c[0])
    c[0].opposite().glue(a)
    c[0].reverse().glue(b)
    c[1].opposite().glue(arrow1.reverse())
    c[1].reverse().glue(arrow2.reverse())
    self.clear_tet(arrow1.Tetrahedron)
    self.clear_tet(arrow2.Tetrahedron)
    self.build_edge_classes()
    if VERBOSE:
        print('0->2')
        print(self.EdgeValences)
    return True