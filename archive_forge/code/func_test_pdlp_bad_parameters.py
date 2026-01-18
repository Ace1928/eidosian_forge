import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def test_pdlp_bad_parameters(self) -> None:
    x = cp.Variable(1)
    prob = cp.Problem(cp.Maximize(x), [x <= 1])
    with self.assertRaises(cp.error.SolverError):
        prob.solve(solver='PDLP', parameters_proto='not a proto')