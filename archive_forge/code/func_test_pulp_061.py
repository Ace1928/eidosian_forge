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
def test_pulp_061(self):
    prob = LpProblem('sample', const.LpMaximize)
    dummy = LpVariable('dummy')
    c1 = LpVariable('c1', 0, 1, const.LpBinary)
    c2 = LpVariable('c2', 0, 1, const.LpBinary)
    prob += dummy
    prob += c1 + c2 == 2
    prob += c1 <= 0
    print('\t Testing another integer infeasible problem')
    if self.solver.__class__ in [GUROBI_CMD, SCIP_CMD, FSCIP_CMD, SCIP_PY]:
        pulpTestCheck(prob, self.solver, [const.LpStatusNotSolved])
    elif self.solver.__class__ in [GLPK_CMD]:
        pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible, const.LpStatusUndefined])
    else:
        pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible])