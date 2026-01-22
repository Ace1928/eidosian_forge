from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def _dihedral_angle(vertex_gram_adjoint, i, j):
    if i == j:
        return 0
    cij = vertex_gram_adjoint[i][j]
    cii = vertex_gram_adjoint[i][i]
    cjj = vertex_gram_adjoint[j][j]
    ciicjj = cii * cjj
    if not ciicjj > 0:
        raise BadDihedralAngleError('cii*cjj not positive')
    t = cij / ciicjj.sqrt()
    if not abs(t) < 1:
        raise BadDihedralAngleError('|cos(angle)| not < 1')
    return t.arccos()