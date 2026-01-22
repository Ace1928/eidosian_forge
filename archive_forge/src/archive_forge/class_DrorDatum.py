import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
class DrorDatum:
    """
    The (omega, A) pair which is the invariant defined in the first column of
    http://www.math.toronto.edu/drorbn/Talks/Aarhus-1507/
    """

    def __init__(self, link, ordered_crossings):
        self.strand_indices = StrandIndices(link, ordered_crossings)
        self.ring = R = PolynomialRing(ZZ, 't').fraction_field()
        self.omega = R.one()
        self.A = matrix(R, 0, 0)

    def add_crossing(self, crossing):
        indices = self.strand_indices
        t = self.ring.gen()
        a, b = entry_pts_ab(crossing)
        n = self.A.nrows()
        assert indices[a] == n and indices[b] == n + 1
        T = t if crossing.sign == 1 else t ** (-1)
        B = matrix([[1, 1 - T], [0, T]])
        self.A = block_diagonal_matrix([self.A, B])

    def merge(self, cs_a, cs_b):
        indices, A = (self.strand_indices, self.A)
        a, b = (indices[cs_a], indices[cs_b])
        if a == b:
            raise ClosedComponentCreated
        mu = 1 - A[a, b]
        self.omega *= mu
        self.A = strand_matrix_merge(A, a, b)
        indices.merge(cs_a, cs_b)