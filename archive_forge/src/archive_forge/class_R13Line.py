from .fixed_points import r13_fixed_points_of_psl2c_matrix # type: ignore
from ..hyperboloid import r13_dot, o13_inverse # type: ignore
from ..upper_halfspace import psl2c_to_o13 # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..sage_helper import _within_sage # type: ignore
class R13Line:
    """
    A line in the hyperboloid model - represented by two
    like-like vectors spanning the line.

    For distance computations, the inner product between the two
    vectors is stored as well.
    """

    def __init__(self, points, inner_product=None):
        """
        inner_product can be given if known, otherwise, will be computed.
        """
        self.points = points
        if inner_product is None:
            self.inner_product = r13_dot(points[0], points[1])
        else:
            self.inner_product = inner_product

    def transformed(self, m):
        """
        Returns image of the line under given O13-matrix m.
        """
        return R13Line([m * point for point in self.points], self.inner_product)