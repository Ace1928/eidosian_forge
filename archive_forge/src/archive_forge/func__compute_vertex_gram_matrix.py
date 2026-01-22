from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def _compute_vertex_gram_matrix(tet, edge_lengths):

    def entry(i, j):
        if i == j:
            return -1
        e = t3m.ZeroSubsimplices[i] | t3m.ZeroSubsimplices[j]
        edge = tet.Class[e]
        return edge_lengths[edge.Index]
    return matrix([[entry(i, j) for j in range(4)] for i in range(4)])