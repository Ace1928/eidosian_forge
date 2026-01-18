import string
from ..sage_helper import _within_sage, sage_method
def univ_matrix_norm(A):
    return max([0] + [univ_abs(a) for a in A.list()])