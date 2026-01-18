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
def test_pulp_013(self):
    prob = LpProblem('test013', const.LpMinimize)
    x = LpVariable('x' * 120, 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0)
    w = LpVariable('w', 0)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7, 'c3')
    prob += (w >= 0, 'c4')
    print('\t Testing Long Names')
    if self.solver.__class__ in [CPLEX_CMD, GLPK_CMD, GUROBI_CMD, MIPCL_CMD, SCIP_CMD, FSCIP_CMD, SCIP_PY, HiGHS, HiGHS_CMD, XPRESS, XPRESS_CMD]:
        try:
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})
        except PulpError:
            pass
    else:
        pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})