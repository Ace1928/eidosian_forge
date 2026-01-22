import os
import numpy as np
from numpy import (asarray, real, imag, conj, zeros, ndarray, concatenate,
from scipy.sparse import coo_matrix, issparse
@staticmethod
def _get_symmetry(a):
    m, n = a.shape
    if m != n:
        return MMFile.SYMMETRY_GENERAL
    issymm = True
    isskew = True
    isherm = a.dtype.char in 'FD'
    if issparse(a):
        a = a.tocoo()
        row, col = a.nonzero()
        if (row < col).sum() != (row > col).sum():
            return MMFile.SYMMETRY_GENERAL
        a = a.todok()

        def symm_iterator():
            for (i, j), aij in a.items():
                if i > j:
                    aji = a[j, i]
                    yield (aij, aji, False)
                elif i == j:
                    yield (aij, aij, True)
    else:

        def symm_iterator():
            for j in range(n):
                for i in range(j, n):
                    aij, aji = (a[i][j], a[j][i])
                    yield (aij, aji, i == j)
    for aij, aji, is_diagonal in symm_iterator():
        if isskew and is_diagonal and (aij != 0):
            isskew = False
        else:
            if issymm and aij != aji:
                issymm = False
            with np.errstate(over='ignore'):
                if isskew and aij != -aji:
                    isskew = False
            if isherm and aij != conj(aji):
                isherm = False
        if not (issymm or isskew or isherm):
            break
    if issymm:
        return MMFile.SYMMETRY_SYMMETRIC
    if isskew:
        return MMFile.SYMMETRY_SKEW_SYMMETRIC
    if isherm:
        return MMFile.SYMMETRY_HERMITIAN
    return MMFile.SYMMETRY_GENERAL