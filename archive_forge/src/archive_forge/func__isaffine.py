from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _isaffine(self):
    if not self._cvxterms and (not self._ccvterms):
        return True
    else:
        return False