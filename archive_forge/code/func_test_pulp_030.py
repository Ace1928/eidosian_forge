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
def test_pulp_030(self):
    prob = LpProblem('test030', const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0, None, const.LpInteger)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7.5, 'c3')
    self.solver.mip = 0
    print('\t Testing MIP relaxation')
    if self.solver.__class__ in [GUROBI_CMD, CHOCO_CMD, MIPCL_CMD, SCIP_CMD, FSCIP_CMD, SCIP_PY]:
        pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 3.0, y: -0.5, z: 7})
    else:
        pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 3.5, y: -1, z: 6.5})