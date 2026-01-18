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
def walk_and_orient(self, tet, sign):
    if tet.Checked == 1:
        return
    tet.Checked = 1
    if sign == 0:
        tet.reverse()
    for ssimp in TwoSubsimplices:
        if tet.Neighbor[ssimp] is not None:
            self.walk_and_orient(tet.Neighbor[ssimp], tet.Gluing[ssimp].sign())