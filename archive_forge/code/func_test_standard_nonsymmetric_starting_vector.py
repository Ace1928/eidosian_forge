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
def test_standard_nonsymmetric_starting_vector():
    params = NonSymmetricParams()
    sigma = None
    symmetric = False
    for k in [1, 2, 3, 4]:
        for d in params.complex_test_cases:
            for typ in 'FD':
                A = d['mat']
                n = A.shape[0]
                v0 = random.rand(n).astype(typ)
                eval_evec(symmetric, d, typ, k, 'LM', v0, sigma)