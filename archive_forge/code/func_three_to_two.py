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
def three_to_two(self, edge_or_arrow, return_arrow=False, must_succeed=False, unsafe_mode=False):
    """
        Replaces the star of an edge of valence 3 by two tetrahedra.

        Options and return value are the same as ``two_to_three``.
        """
    edge, a = edge_and_arrow(edge_or_arrow)
    if not unsafe_mode:
        possible, reason = self._edge_permits_three_to_two(edge)
        if not possible:
            if must_succeed:
                raise ValueError(reason)
            return False
    a_orig = a.copy()
    b = self.new_arrow()
    c = self.new_arrow()
    b.glue(c)
    b_orig = b.copy()
    b.reverse()
    b_to_return = b.copy()
    for i in range(3):
        b.glue(a.opposite().glued())
        c.glue(a.reverse().glued())
        b.rotate(-1)
        c.rotate(1)
        a.reverse().opposite().next()
    self._three_to_two_move_hook(a_orig, (b_orig, b, c))
    if unsafe_mode:
        tet0 = a_orig.Tetrahedron
        tet1 = a_orig.next().Tetrahedron
        tet2 = a_orig.next().Tetrahedron
        self.delete_tet(tet0)
        self.delete_tet(tet1)
        self.delete_tet(tet2)
    else:
        for corner in edge.Corners:
            self.delete_tet(corner.Tetrahedron)
    if not unsafe_mode:
        self.build_edge_classes()
    if VERBOSE:
        print('3->2')
        print(self.EdgeValences)
    if return_arrow:
        return b_to_return
    return True