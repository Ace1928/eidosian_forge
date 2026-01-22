import string
from ..sage_helper import _within_sage, sage_method
class PhiAlpha:

    def __init__(self, phi, alpha):
        self.base_ring = phi.range()
        self.image_ring = MatrixSpace(self.base_ring, 2)
        self.phi, self.alpha = (phi, alpha)

    def range(self):
        return self.image_ring

    def __call__(self, word):
        a = self.phi(word)
        A = self.alpha(word)
        M = self.image_ring(0)
        M[0, 0], M[0, 1], M[1, 0], M[1, 1] = (a * A[0, 0], a * A[0, 1], a * A[1, 0], a * A[1, 1])
        return M