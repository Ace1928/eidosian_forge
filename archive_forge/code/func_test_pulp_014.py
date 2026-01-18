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
def test_pulp_014(self):
    prob = LpProblem('test014', const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('x', -1, 1)
    z = LpVariable('z', 0)
    w = LpVariable('w', 0)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7, 'c3')
    prob += (w >= 0, 'c4')
    print('\t Testing repeated Names')
    if self.solver.__class__ in [COIN_CMD, COINMP_DLL, PULP_CBC_CMD, CPLEX_CMD, CPLEX_PY, GLPK_CMD, GUROBI_CMD, CHOCO_CMD, MIPCL_CMD, MOSEK, SCIP_CMD, FSCIP_CMD, SCIP_PY, HiGHS, HiGHS_CMD, XPRESS, XPRESS_CMD, XPRESS_PY]:
        try:
            pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})
        except PulpError:
            pass
    else:
        pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})