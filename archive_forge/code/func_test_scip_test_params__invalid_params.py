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
def test_scip_test_params__invalid_params(self) -> None:
    prob = self.get_simple_problem()
    with pytest.raises(KeyError) as ke:
        prob.solve(solver='SCIP', a='what?')
        exc = "One or more solver params in ['a'] are not valid: 'Not a valid parameter name'"
        assert ke.exception == exc