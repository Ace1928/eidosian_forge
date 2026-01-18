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
def tile_until_done(self):
    while not self.is_done():
        self.process_next_unglued_generator()