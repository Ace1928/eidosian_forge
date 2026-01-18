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
def test_pulp_001(self):
    """
            Test that a variable is deleted when it is suptracted to 0
            """
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0)
    c1 = x + y <= 5
    c2 = c1 + z - z
    print('\t Testing zero subtraction')
    assert str(c2)