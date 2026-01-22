from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
def _abs_sqr(z):
    return z.real() ** 2 + z.imag() ** 2