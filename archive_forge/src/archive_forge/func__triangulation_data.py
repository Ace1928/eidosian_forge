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
def _triangulation_data(self):
    ans = []
    tet_to_index = {T: i for i, T in enumerate(self.Tetrahedra)}
    for T in self.Tetrahedra:
        neighbors, perms = ([], [])
        for v in TwoSubsimplices:
            if T.Neighbor[v] is None:
                neighbor, perm = (None, None)
            else:
                neighbor = tet_to_index[T.Neighbor[v]]
                perm = T.Gluing[v].tuple()
            neighbors.append(neighbor)
            perms.append(perm)
        ans.append((neighbors, perms))
    return ans