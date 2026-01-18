import unittest
import numpy as np
import pytest
import scipy
import scipy.sparse as sp
import scipy.stats
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize
def test_trace_sign_psd(self) -> None:
    """Test sign of trace for psd/nsd inputs.
        """
    X_psd = cp.Variable((2, 2), PSD=True)
    X_nsd = cp.Variable((2, 2), NSD=True)
    psd_trace = cp.trace(X_psd)
    nsd_trace = cp.trace(X_nsd)
    assert psd_trace.is_nonneg()
    assert nsd_trace.is_nonpos()