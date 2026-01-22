import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
class TestEighCases(HermitianTestCase, HermitianGeneralizedTestCase):

    def do(self, a, b, tags):
        res = linalg.eigh(a)
        ev, evc = (res.eigenvalues, res.eigenvectors)
        evalues, evectors = linalg.eig(a)
        evalues.sort(axis=-1)
        assert_almost_equal(ev, evalues)
        assert_allclose(dot_generalized(a, evc), np.asarray(ev)[..., None, :] * np.asarray(evc), rtol=get_rtol(ev.dtype))
        ev2, evc2 = linalg.eigh(a, 'U')
        assert_almost_equal(ev2, evalues)
        assert_allclose(dot_generalized(a, evc2), np.asarray(ev2)[..., None, :] * np.asarray(evc2), rtol=get_rtol(ev.dtype), err_msg=repr(a))