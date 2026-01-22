from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
class RealHoroTriangle:
    """
    A horosphere cross section in the corner of an ideal tetrahedron.
    The sides of the triangle correspond to faces of the tetrahedron.
    The lengths stored for the triangle are real.
    """

    def __init__(self, tet, vertex, known_side, length_of_side):
        left_side, center_side, right_side, z_left, z_right = HoroTriangleBase._sides_and_cross_ratios(tet, vertex, known_side)
        L = length_of_side
        self.lengths = {center_side: L, left_side: abs(z_left) * L, right_side: L / abs(z_right)}
        a, b, c = self.lengths.values()
        self.area = L * L * z_left.imag() / 2
        self.circumradius = a * b * c / (4 * self.area)

    def rescale(self, t):
        """Rescales the triangle by a Euclidean dilation"""
        for face in self.lengths:
            self.lengths[face] *= t
        self.circumradius *= t
        self.area *= t * t

    @staticmethod
    def direction_sign():
        return +1