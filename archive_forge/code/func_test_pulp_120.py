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
def test_pulp_120(self):
    """
            Test the ability to use Elastic constraints
            """
    prob = LpProblem('test120', const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0)
    w = LpVariable('w')
    prob += (x + 4 * y + 9 * z + w, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7, 'c3')
    prob.extend((w >= -1).makeElasticSubProblem())
    print('\t Testing elastic constraints (no change)')
    pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: -1})