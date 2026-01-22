from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def _compute_eta(self, tet_and_perm):
    tet_index, perm = tet_and_perm
    vertex_gram_matrix = self.vertex_gram_matrices[tet_index]
    vij = vertex_gram_matrix[perm[0], perm[1]]
    vik = vertex_gram_matrix[perm[0], perm[2]]
    vjk = vertex_gram_matrix[perm[1], perm[2]]
    t = (vij * vik + vjk) / ((vij ** 2 - 1) * (vik ** 2 - 1)).sqrt()
    return t.arccos()