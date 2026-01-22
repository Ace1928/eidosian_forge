from .fixed_points import r13_fixed_points_of_psl2c_matrix # type: ignore
from ..hyperboloid import r13_dot, o13_inverse # type: ignore
from ..upper_halfspace import psl2c_to_o13 # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..sage_helper import _within_sage # type: ignore
class R13LineWithMatrix:
    """
    A line in the hyperboloid model together with a O(1,3)-matrix fixing
    the line (set-wise).
    """

    def __init__(self, r13_line: R13Line, o13_matrix):
        self.r13_line = r13_line
        self.o13_matrix = o13_matrix

    @staticmethod
    def from_psl2c_matrix(m):
        """
        Given a loxodromic PSL(2,C)-matrix m, returns the line (together
        with the O(1,3)-matrix corresponding to m) fixed by m in
        the hyperboloid model.
        """
        return R13LineWithMatrix(R13Line(r13_fixed_points_of_psl2c_matrix(m)), psl2c_to_o13(m))

    def transformed(self, m):
        """
        Returns image of line with matrix under given O13-matrix m.

        That is, the matrix will be conjugated by m so that the new
        matrix will fix the image of the line (set-wise).
        """
        return R13LineWithMatrix(self.r13_line.transformed(m), m * self.o13_matrix * o13_inverse(m))