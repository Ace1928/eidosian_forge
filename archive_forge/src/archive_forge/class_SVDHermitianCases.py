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
class SVDHermitianCases(HermitianTestCase, HermitianGeneralizedTestCase):

    def do(self, a, b, tags):
        u, s, vt = linalg.svd(a, False, hermitian=True)
        assert_allclose(a, dot_generalized(np.asarray(u) * np.asarray(s)[..., None, :], np.asarray(vt)), rtol=get_rtol(u.dtype))

        def hermitian(mat):
            axes = list(range(mat.ndim))
            axes[-1], axes[-2] = (axes[-2], axes[-1])
            return np.conj(np.transpose(mat, axes=axes))
        assert_almost_equal(np.matmul(u, hermitian(u)), np.broadcast_to(np.eye(u.shape[-1]), u.shape))
        assert_almost_equal(np.matmul(vt, hermitian(vt)), np.broadcast_to(np.eye(vt.shape[-1]), vt.shape))
        assert_equal(np.sort(s)[..., ::-1], s)
        assert_(consistent_subclass(u, a))
        assert_(consistent_subclass(vt, a))