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
def test_pulp_022(self):
    prob = LpProblem('test022', const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0, None, const.LpInteger)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7.5, 'c3')
    x.setInitialValue(3)
    y.setInitialValue(-0.5)
    z.setInitialValue(7)
    if self.solver.name in ['GUROBI', 'GUROBI_CMD', 'CPLEX_CMD', 'CPLEX_PY', 'COPT', 'HiGHS_CMD']:
        self.solver.optionsDict['warmStart'] = True
    print('\t Testing Initial value in MIP solution')
    pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 3, y: -0.5, z: 7})