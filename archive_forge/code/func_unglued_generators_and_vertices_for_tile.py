from ...sage_helper import _within_sage
from ...math_basics import correct_max
from ...snap.kernel_structures import *
from ...snap.fundamental_polyhedron import *
from ...snap.mcomplex_base import *
from ...snap.t3mlite import simplex
from ...snap import t3mlite as t3m
from ...exceptions import InsufficientPrecisionError
from ..cuspCrossSection import ComplexCuspCrossSection
from ..upper_halfspace.ideal_point import *
from ..interval_tree import *
from .cusp_translate_engine import *
import heapq
def unglued_generators_and_vertices_for_tile(self, tile):
    unglued_generators = []
    unglued_vertices = set(self.mcomplex.Vertices)
    for g, gen_m in sorted(self.mcomplex.GeneratorMatrices.items()):
        other_m = tile.matrix * gen_m
        other_tile = self.find_tile(other_m)
        if other_tile:
            if tile is not other_tile:
                for (corner, other_corner), perm in self.mcomplex.Generators[g]:
                    for v in simplex.VerticesOfFaceCounterclockwise[corner.Subsimplex]:
                        vertex = corner.Tetrahedron.Class[v]
                        unglued_vertices.discard(vertex)
        else:
            unglued_generators.append(g)
    return (unglued_generators, unglued_vertices)