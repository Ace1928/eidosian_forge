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
def test_pulp_009(self):
    prob = LpProblem('test09', const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0)
    w = LpVariable('w', 0)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (lpSum([v for v in [x] if False]) >= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7, 'c3')
    prob += (w >= 0, 'c4')
    print('\t Testing inconsistent lp solution')
    if self.solver.__class__ in [PULP_CBC_CMD, COIN_CMD]:
        pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible], {x: 4, y: -1, z: 6, w: 0}, use_mps=False)
    elif self.solver.__class__ in [CHOCO_CMD, MIPCL_CMD]:
        pass
    else:
        pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible, const.LpStatusNotSolved, const.LpStatusUndefined])