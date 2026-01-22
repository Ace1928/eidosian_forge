from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def add_vertex_positions(self, vertex, edge, position):
    """
        Adds a dictionary vertex_positions mapping
        an edge (such as t3m.simplex.E01) to complex position
        for the vertex of the horotriangle obtained by
        intersecting the edge with the horosphere.

        Two of these positions are computed from the one given
        using the complex edge lengths. The given vertex and
        edge are t3m-style.
        """
    self.vertex_positions = {}
    vertex_link = _face_edge_face_triples_for_vertex_link[vertex]
    for i in range(3):
        if edge == vertex_link[i][1]:
            break
    for j in range(3):
        face0, edge, face1 = vertex_link[(i + j) % 3]
        self.vertex_positions[edge] = position
        position += self.lengths[face1]