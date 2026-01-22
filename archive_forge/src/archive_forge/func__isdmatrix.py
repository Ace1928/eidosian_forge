from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _isdmatrix(a):
    """ True if a is a nonempty dense 'd' matrix. """
    if type(a) is matrix and a.typecode == 'd' and (min(a.size) != 0):
        return True
    else:
        return False