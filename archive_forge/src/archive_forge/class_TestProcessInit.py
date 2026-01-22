import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
import numpy as np
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.interior_point.interior_point import (
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
class TestProcessInit(unittest.TestCase):

    def testprocess_init(self):
        lb = np.array([-np.inf, -np.inf, -2, -2], dtype=np.double)
        ub = np.array([np.inf, 2, np.inf, 2], dtype=np.double)
        x = np.array([0, 0, 0, 0], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([0, 0, 0, 0], dtype=np.double)))
        x = np.array([-2, -2, -2, -2], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([-2, -2, -1, 0], dtype=np.double)))
        x = np.array([-3, -3, -3, -3], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([-3, -3, -1, 0], dtype=np.double)))
        x = np.array([2, 2, 2, 2], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([2, 1, 2, 0], dtype=np.double)))
        x = np.array([3, 3, 3, 3], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([3, 1, 3, 0], dtype=np.double)))

    def testprocess_init_duals(self):
        x = np.array([0, 0, 0, 0], dtype=np.double)
        lb = np.array([-5, 0, -np.inf, 2], dtype=np.double)
        process_init_duals_lb(x, lb)
        self.assertTrue(np.allclose(x, np.array([1, 1, 0, 1], dtype=np.double)))
        x = np.array([-1, -1, -1, -1], dtype=np.double)
        process_init_duals_lb(x, lb)
        self.assertTrue(np.allclose(x, np.array([1, 1, 0, 1], dtype=np.double)))
        x = np.array([2, 2, 2, 2], dtype=np.double)
        ub = np.array([-5, 0, np.inf, 2], dtype=np.double)
        process_init_duals_ub(x, ub)
        self.assertTrue(np.allclose(x, np.array([2, 2, 0, 2], dtype=np.double)))