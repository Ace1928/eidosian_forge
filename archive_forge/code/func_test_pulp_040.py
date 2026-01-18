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
def test_pulp_040(self):
    prob = LpProblem('test040', const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0, None, const.LpInteger)
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7.5, 'c3')
    print('\t Testing feasibility problem (no objective)')
    pulpTestCheck(prob, self.solver, [const.LpStatusOptimal])