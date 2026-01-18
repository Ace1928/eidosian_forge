import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
from ..pyomo_nlp import PyomoGreyBoxNLP
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
def test_pressure_drop_two_equalities_with_hessian(self):
    egbm = ex_models.PressureDropTwoEqualitiesWithHessian()
    input_names = egbm.input_names()
    self.assertEqual(input_names, ['Pin', 'c', 'F', 'P2', 'Pout'])
    eq_con_names = egbm.equality_constraint_names()
    self.assertEqual(eq_con_names, ['pdrop2', 'pdropout'])
    output_names = egbm.output_names()
    self.assertEqual([], output_names)
    egbm.set_input_values(np.asarray([100, 2, 3, 20, 50], dtype=np.float64))
    egbm.set_equality_constraint_multipliers(np.asarray([3, 5], dtype=np.float64))
    egbm.set_output_constraint_multipliers(np.asarray([]))
    with self.assertRaises(AssertionError):
        egbm.set_output_constraint_multipliers(np.asarray([1], dtype=np.float64))
    eq = egbm.evaluate_equality_constraints()
    self.assertTrue(np.array_equal(eq, np.asarray([-44, 66], dtype=np.float64)))
    with self.assertRaises(NotImplementedError):
        tmp = egbm.evaluate_outputs()
    with self.assertRaises(NotImplementedError):
        tmp = egbm.evaluate_jacobian_outputs()
    jac_eq = egbm.evaluate_jacobian_equality_constraints()
    self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)))
    self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)))
    self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 18, 24, 1, 18, 24, -1, 1], dtype=np.float64)))
    with self.assertRaises(AttributeError):
        hess_outputs = egbm.evaluate_hessian_outputs()
    hess = egbm.evaluate_hessian_equality_constraints()
    self.assertTrue(np.array_equal(hess.row, np.asarray([2, 2], dtype=np.int64)))
    self.assertTrue(np.array_equal(hess.col, np.asarray([1, 2], dtype=np.int64)))
    self.assertTrue(np.array_equal(hess.data, np.asarray([96.0, 64.0], dtype=np.float64)))