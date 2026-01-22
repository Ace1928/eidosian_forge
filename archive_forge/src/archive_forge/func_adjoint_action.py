import string
from ..sage_helper import _within_sage, sage_method
def adjoint_action(A):
    a, b, c, d = A.list()
    return matrix([[a ** 2, 2 * a * b, b ** 2], [a * c, b * c + a * d, b * d], [c ** 2, 2 * c * d, d ** 2]])