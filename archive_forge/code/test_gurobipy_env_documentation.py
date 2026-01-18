import unittest
from pulp import GUROBI, LpProblem, LpVariable, const

        Check that we cannot initialise environments after a memory leak. On a
        single-use license this passes (fails to initialise a dummy env with a
        memory leak).
        