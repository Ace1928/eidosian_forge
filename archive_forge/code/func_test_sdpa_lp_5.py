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
def test_sdpa_lp_5(self) -> None:
    StandardTestLPs.test_lp_5(solver='SDPA', betaBar=0.1, gammaStar=0.8, epsilonDash=8e-06)