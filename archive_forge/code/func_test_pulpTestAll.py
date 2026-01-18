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
def test_pulpTestAll(self):
    """
            Test the availability of the function pulpTestAll
            """
    print('\t Testing the availability of the function pulpTestAll')
    from pulp import pulpTestAll