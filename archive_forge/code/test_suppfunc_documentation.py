import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest

    Test the implementation of support function atoms.

    Relevant source code includes:
        cvxpy.atoms.suppfunc
        cvxpy.transforms.suppfunc
        cvxpy.reductions.dcp2cone.canonicalizers.suppfunc_canon
    