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
def test_pulp_018(self):
    prob = LpProblem('test018', const.LpMinimize)
    x = LpVariable('x' * 90, 0, 4)
    y = LpVariable('y' * 90, -1, 1)
    z = LpVariable('z' * 90, 0)
    w = LpVariable('w' * 90, 0)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7, 'c3')
    prob += (w >= 0, 'c4')
    if self.solver.__class__ in [PULP_CBC_CMD, COIN_CMD]:
        print('\t Testing Long lines in LP')
        pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0}, use_mps=False)