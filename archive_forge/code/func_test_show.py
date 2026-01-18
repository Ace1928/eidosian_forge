import itertools
import platform
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
@pytest.mark.parametrize('case', IterativeParams().cases)
def test_show(case, capsys):

    def cb(x):
        pass
    x, info = tfqmr(case.A, case.b, callback=cb, show=True)
    out, err = capsys.readouterr()
    if case.name == 'sym-nonpd':
        exp = ''
    elif case.name in ('nonsymposdef', 'nonsymposdef-F'):
        exp = 'TFQMR: Linear solve not converged due to reach MAXIT iterations'
    else:
        exp = 'TFQMR: Linear solve converged due to reach TOL iterations'
    assert out.startswith(exp)
    assert err == ''