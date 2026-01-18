from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_problem_root(self):
    for f in [F, F2, F2_lucky, F3, F4_powell, F5, F6]:
        for meth in SOLVERS:
            if meth in f.KNOWN_BAD:
                if meth in MUST_WORK:
                    self._check_func_fail(f, meth)
                continue
            self._check_root(f, meth)