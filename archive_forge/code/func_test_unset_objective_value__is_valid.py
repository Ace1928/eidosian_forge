import os
import tempfile
from pulp.constants import PulpError
from pulp.apis import *
from pulp import LpVariable, LpProblem, lpSum, LpConstraintVar, LpFractionConstraint
from pulp import constants as const
from pulp.tests.bin_packing_problem import create_bin_packing_problem
from pulp.utilities import makeDict
import functools
import unittest
def test_unset_objective_value__is_valid(self):
    """Given a valid problem that does not converge,
            assert that it is still categorised as valid.
            """
    name = self._testMethodName
    prob = LpProblem(name, const.LpMaximize)
    x = LpVariable('x')
    prob += 0 * x
    prob += x >= 1
    pulpTestCheck(prob, self.solver, [const.LpStatusOptimal])
    self.assertTrue(prob.valid())