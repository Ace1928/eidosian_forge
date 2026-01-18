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
def upper_bound_for_height_of_unglued_generator(self, tile, g):
    heights = [tile.height_of_face(corner).upper() for (corner, other_corner), perm in self.mcomplex.Generators[g]]
    for height in heights:
        if height.is_NaN():
            raise InsufficientPrecisionError('A NaN occurred when computing the height of a triangle face. This can most likely be avoided by increasing the precision.')
    return max(heights)