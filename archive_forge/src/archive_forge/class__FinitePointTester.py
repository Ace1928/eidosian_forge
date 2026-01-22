from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
class _FinitePointTester:
    """
    A test rig for FinitePoint.

    Run the test rig::

        sage: _FinitePointTester().run_tests()

    """

    def matrix1(self):
        from sage.all import RIF, CIF, matrix
        return matrix([[CIF(RIF(1.3), RIF(-0.4)), CIF(RIF(5.6), RIF(2.3))], [CIF(RIF(-0.3), RIF(0.1)), CIF(1)]])

    def extended_matrix1(self, isOrientationReversing):
        return ExtendedMatrix(self.matrix1(), isOrientationReversing)

    def matrix2(self):
        from sage.all import RIF, CIF, matrix
        return matrix([[CIF(RIF(0.3), RIF(-1.4)), CIF(RIF(3.6), RIF(6.3))], [CIF(RIF(-0.3), RIF(1.1)), CIF(1)]])

    def extended_matrix2(self, isOrientationReversing):
        return ExtendedMatrix(self.matrix2(), isOrientationReversing)

    def images_have_same_distance(self, m):
        from sage.rings.real_mpfi import RealIntervalFieldElement
        from sage.all import RIF, CIF
        a = FinitePoint(CIF(RIF(3.5), RIF(-3.0)), RIF(8.5))
        b = FinitePoint(CIF(RIF(4.5), RIF(-4.5)), RIF(9.6))
        d_before = a.dist(b)
        a = a.translate_PGL(m)
        b = b.translate_PGL(m)
        d_after = a.dist(b)
        if not isinstance(d_before, RealIntervalFieldElement):
            raise Exception('Expected distance to be RIF')
        if not isinstance(d_after, RealIntervalFieldElement):
            raise Exception('Expected distance to be RIF')
        if not abs(d_before - d_after) < RIF(1e-12):
            raise Exception('Distance changed %r %r' % (d_before, d_after))

    def matrix_multiplication_works(self, matrices):
        from sage.all import RIF, CIF, prod
        a = FinitePoint(CIF(RIF(3.5), RIF(-3.0)), RIF(8.5))
        a0 = a.translate_PGL(prod(matrices))
        for m in matrices[::-1]:
            a = a.translate_PGL(m)
        if not a.dist(a0) < RIF(1e-06):
            raise Exception('Distance %r' % a.dist(a0))

    def run_tests(self):
        m1o = self.extended_matrix1(False)
        m1r = self.extended_matrix1(True)
        m2o = self.extended_matrix2(False)
        m2r = self.extended_matrix2(True)
        self.images_have_same_distance(m1o)
        self.images_have_same_distance(m1o * m1o)
        self.images_have_same_distance(m1o * m1o * m1o)
        self.images_have_same_distance(m1o * m1o * m1o * m1o)
        self.images_have_same_distance(m1o * m2o)
        self.matrix_multiplication_works([m1o, m1o, m2r, m1o])
        self.matrix_multiplication_works([m1o, m1r, m2r, m1o])
        self.matrix_multiplication_works([m2r, m1o, m2r, m1o])