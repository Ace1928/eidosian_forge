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
def test_pulp_110(self):
    """
            Test the ability to use fractional constraints
            """
    prob = LpProblem('test110', const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0)
    w = LpVariable('w', 0)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7, 'c3')
    prob += (w >= 0, 'c4')
    prob += LpFractionConstraint(x, z, const.LpConstraintEQ, 0.5, name='c5')
    print('\t Testing fractional constraints')
    pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], {x: 10 / 3.0, y: -1 / 3.0, z: 20 / 3.0, w: 0})