import threading
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
def argsort_which(eigenvalues, typ, k, which, sigma=None, OPpart=None, mode=None):
    """Return sorted indices of eigenvalues using the "which" keyword
    from eigs and eigsh"""
    if sigma is None:
        reval = np.round(eigenvalues, decimals=_ndigits[typ])
    else:
        if mode is None or mode == 'normal':
            if OPpart is None:
                reval = 1.0 / (eigenvalues - sigma)
            elif OPpart == 'r':
                reval = 0.5 * (1.0 / (eigenvalues - sigma) + 1.0 / (eigenvalues - np.conj(sigma)))
            elif OPpart == 'i':
                reval = -0.5j * (1.0 / (eigenvalues - sigma) - 1.0 / (eigenvalues - np.conj(sigma)))
        elif mode == 'cayley':
            reval = (eigenvalues + sigma) / (eigenvalues - sigma)
        elif mode == 'buckling':
            reval = eigenvalues / (eigenvalues - sigma)
        else:
            raise ValueError("mode='%s' not recognized" % mode)
        reval = np.round(reval, decimals=_ndigits[typ])
    if which in ['LM', 'SM']:
        ind = np.argsort(abs(reval))
    elif which in ['LR', 'SR', 'LA', 'SA', 'BE']:
        ind = np.argsort(np.real(reval))
    elif which in ['LI', 'SI']:
        if typ.islower():
            ind = np.argsort(abs(np.imag(reval)))
        else:
            ind = np.argsort(np.imag(reval))
    else:
        raise ValueError("which='%s' is unrecognized" % which)
    if which in ['LM', 'LA', 'LR', 'LI']:
        return ind[-k:]
    elif which in ['SM', 'SA', 'SR', 'SI']:
        return ind[:k]
    elif which == 'BE':
        return np.concatenate((ind[:k // 2], ind[k // 2 - k:]))