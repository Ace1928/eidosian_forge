from ... import sage_helper
from .. import t3mlite as t3m
from . import link, dual_cellulation
class PeripheralOneCocycle:
    """
    Let M be an ideal triangulation with one cusp, and consider the
    induced triangulation T of the cusp torus.  This object is a
    1-cocycles on T, whose weights are accessed via

    self[tet_num, face_index, vertex_in_face].
    """

    def __init__(self, dual_cellulation_cocycle):
        self.cocycle = dual_cellulation_cocycle
        self.dual_cellulation = D = dual_cellulation_cocycle.cellulation
        self.cusp_triangulation = T = D.dual_triangulation
        self.mcomplex = T.parent_triangulation

    def __getitem__(self, tet_face_vertex):
        tet_num, F, V = tet_face_vertex
        tet = self.mcomplex.Tetrahedra[tet_num]
        triangle = tet.CuspCorners[V]
        for side in triangle.oriented_sides():
            E0, E1 = [link.TruncatedSimplexCorners[V][v] for v in side.vertices]
            if E0 | E1 == F:
                break
        assert E0 | E1 == F
        global_edge = side.edge()
        dual_edge = self.dual_cellulation.from_original[global_edge]
        w = self.cocycle.weights[dual_edge.index]
        s = global_edge.orientation_with_respect_to(side)
        return w * s