import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def tv_inpainting():
    Ucorr = known * Uorig
    variables = []
    constraints = []
    for i in range(colors):
        U = cp.Variable(shape=(rows, cols))
        variables.append(U)
        constraints.append(cp.multiply(known[:, :, i], U) == cp.multiply(known[:, :, i], Ucorr[:, :, i]))
    problem = cp.Problem(cp.Minimize(cp.tv(*variables)), constraints)
    problem.get_problem_data(cp.SCS)