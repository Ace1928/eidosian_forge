import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
Regression test for https://github.com/cvxpy/cvxpy/issues/1668

        Pruning matrices caused order-of-magnitude slow downs in compilation times.
        