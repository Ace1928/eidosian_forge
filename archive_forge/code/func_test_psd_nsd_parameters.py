import warnings
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.wraps import (
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities.linalg import gershgorin_psd_check
def test_psd_nsd_parameters(self) -> None:
    np.random.seed(42)
    a = np.random.normal(size=(100, 95))
    a2 = a.dot(a.T)
    p = Parameter((100, 100), PSD=True)
    p.value = a2
    self.assertItemsAlmostEqual(p.value, a2, places=10)
    m, n = (10, 5)
    A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    A = np.dot(A.T.conj(), A)
    A = np.vstack([np.hstack([np.real(A), -np.imag(A)]), np.hstack([np.imag(A), np.real(A)])])
    p = Parameter(shape=(2 * n, 2 * n), PSD=True)
    p.value = A
    self.assertItemsAlmostEqual(p.value, A)
    p = Parameter(shape=(2, 2), PSD=True)
    self.assertTrue((2 * p).is_psd())
    self.assertTrue((p + p).is_psd())
    self.assertTrue((-p).is_nsd())
    self.assertTrue((-2 * -p).is_psd())
    n = 5
    P = Parameter(shape=(n, n), PSD=True)
    N = Parameter(shape=(n, n), NSD=True)
    np.random.randn(0)
    U = np.random.randn(n, n)
    U = U @ U.T
    evals, U = np.linalg.eigh(U)
    v1 = np.array([3, 2, 1, 1e-08, -1])
    v2 = np.array([3, 2, 2, 1e-06, -1])
    v3 = np.array([3, 2, 2, 0.0001, -1e-06])
    v4 = np.array([-1, 3, 0, 0, 0])
    vs = [v1, v2, v3, v4]
    for vi in vs:
        with self.assertRaises(Exception) as cm:
            P.value = U @ np.diag(vi) @ U.T
        self.assertEqual(str(cm.exception), 'Parameter value must be positive semidefinite.')
        with self.assertRaises(Exception) as cm:
            N.value = -U @ np.diag(vi) @ U.T
        self.assertEqual(str(cm.exception), 'Parameter value must be negative semidefinite.')