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
def test_pulp_080(self):
    """
            Test the reporting of dual variables slacks and reduced costs
            """
    prob = LpProblem('test080', const.LpMinimize)
    x = LpVariable('x', 0, 5)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0)
    c1 = x + y <= 5
    c2 = x + z >= 10
    c3 = -y + z == 7
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (c1, 'c1')
    prob += (c2, 'c2')
    prob += (c3, 'c3')
    if self.solver.__class__ in [CPLEX_CMD, COINMP_DLL, PULP_CBC_CMD, YAPOSIB, PYGLPK]:
        print('\t Testing dual variables and slacks reporting')
        pulpTestCheck(prob, self.solver, [const.LpStatusOptimal], sol={x: 4, y: -1, z: 6}, reducedcosts={x: 0, y: 12, z: 0}, duals={'c1': 0, 'c2': 1, 'c3': 8}, slacks={'c1': 2, 'c2': 0, 'c3': 0})