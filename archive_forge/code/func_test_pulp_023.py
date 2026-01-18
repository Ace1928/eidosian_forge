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
def test_pulp_023(self):
    prob = LpProblem('test023', const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0, None, const.LpInteger)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7.5, 'c3')
    solution = {x: 4, y: -0.5, z: 7}
    for v in [x, y, z]:
        v.setInitialValue(solution[v])
        v.fixValue()
    self.solver.optionsDict['warmStart'] = True
    print('\t Testing fixing value in MIP solution')
    pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], solution)