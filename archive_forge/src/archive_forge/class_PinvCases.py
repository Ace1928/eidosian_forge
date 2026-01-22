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
class PinvCases(LinalgSquareTestCase, LinalgNonsquareTestCase, LinalgGeneralizedSquareTestCase, LinalgGeneralizedNonsquareTestCase):

    def do(self, a, b, tags):
        a_ginv = linalg.pinv(a)
        dot = dot_generalized
        assert_almost_equal(dot(dot(a, a_ginv), a), a, single_decimal=5, double_decimal=11)
        assert_(consistent_subclass(a_ginv, a))