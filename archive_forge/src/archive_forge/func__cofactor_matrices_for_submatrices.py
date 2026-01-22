from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def _cofactor_matrices_for_submatrices(m):

    def cofactor_matrix(r, c):
        submatrix = m.matrix_from_rows_and_columns([k for k in range(4) if k != r], [k for k in range(4) if k != c])
        return submatrix.adjugate().transpose() * (-1) ** (r + c)
    return [[cofactor_matrix(r, c) for c in range(4)] for r in range(4)]