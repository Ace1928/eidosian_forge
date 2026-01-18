import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND

        Tests nonpos flag
        Reference values via MOSEK
        Version: 10.0.46
        