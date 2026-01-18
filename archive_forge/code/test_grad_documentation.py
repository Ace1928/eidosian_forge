from __future__ import division
import numpy as np
import cvxpy as cp
from cvxpy import Maximize, Minimize, Problem
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import linearize
from cvxpy.transforms.partial_optimize import partial_optimize
Test grad for affine atoms.
        