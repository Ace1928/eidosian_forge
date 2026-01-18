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
def test_pulp_090(self):
    prob = LpProblem('test090', const.LpMinimize)
    obj = LpConstraintVar('obj')
    a = LpConstraintVar('C1', const.LpConstraintLE, 5)
    b = LpConstraintVar('C2', const.LpConstraintGE, 10)
    c = LpConstraintVar('C3', const.LpConstraintEQ, 7)
    prob.setObjective(obj)
    prob += a
    prob += b
    prob += c
    prob.setSolver(self.solver)
    x = LpVariable('x', 0, 4, const.LpContinuous, obj + a + b)
    y = LpVariable('y', -1, 1, const.LpContinuous, 4 * obj + a - c)
    prob.resolve()
    z = LpVariable('z', 0, None, const.LpContinuous, 9 * obj + b + c)
    if self.solver.__class__ in [COINMP_DLL]:
        print('\t Testing resolve of problem')
        prob.resolve()