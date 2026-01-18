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
def test_unbounded_problem__is_not_valid(self):
    """Given an unbounded problem, where x will tend to infinity
            to maximise the objective, assert that it is categorised
            as invalid."""
    name = self._testMethodName
    prob = LpProblem(name, const.LpMaximize)
    x = LpVariable('x')
    prob += 1000 * x
    prob += x >= 1
    self.assertFalse(prob.valid())