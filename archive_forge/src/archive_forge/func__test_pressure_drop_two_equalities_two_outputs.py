import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
def _test_pressure_drop_two_equalities_two_outputs(self, ex_model, hessian_support):
    m = pyo.ConcreteModel()
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_model)
    m.egb.inputs['Pin'].value = 100
    m.egb.inputs['Pin'].setlb(50)
    m.egb.inputs['Pin'].setub(150)
    m.egb.inputs['c'].value = 2
    m.egb.inputs['c'].setlb(1)
    m.egb.inputs['c'].setub(5)
    m.egb.inputs['F'].value = 3
    m.egb.inputs['F'].setlb(1)
    m.egb.inputs['F'].setub(5)
    m.egb.inputs['P1'].value = 80
    m.egb.inputs['P1'].setlb(10)
    m.egb.inputs['P1'].setub(90)
    m.egb.inputs['P3'].value = 70
    m.egb.inputs['P3'].setlb(20)
    m.egb.inputs['P3'].setub(80)
    m.egb.outputs['P2'].value = 75
    m.egb.outputs['P2'].setlb(15)
    m.egb.outputs['P2'].setub(85)
    m.egb.outputs['Pout'].value = 50
    m.egb.outputs['Pout'].setlb(30)
    m.egb.outputs['Pout'].setub(70)
    m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
    pyomo_nlp = PyomoNLPWithGreyBoxBlocks(m)
    self.assertEqual(7, pyomo_nlp.n_primals())
    self.assertEqual(4, pyomo_nlp.n_constraints())
    self.assertEqual(16, pyomo_nlp.nnz_jacobian())
    if hessian_support:
        self.assertEqual(4, pyomo_nlp.nnz_hessian_lag())
    comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]', 'egb.inputs[P1]', 'egb.inputs[P3]', 'egb.outputs[P2]', 'egb.outputs[Pout]']
    x_order = pyomo_nlp.primals_names()
    comparison_c_order = ['egb.pdrop1', 'egb.pdrop3', 'egb.output_constraints[P2]', 'egb.output_constraints[Pout]']
    c_order = pyomo_nlp.constraint_names()
    xlb = pyomo_nlp.primals_lb()
    comparison_xlb = np.asarray([50, 1, 1, 10, 20, 15, 30], dtype=np.float64)
    check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
    xub = pyomo_nlp.primals_ub()
    comparison_xub = np.asarray([150, 5, 5, 90, 80, 85, 70], dtype=np.float64)
    check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
    clb = pyomo_nlp.constraints_lb()
    comparison_clb = np.asarray([0, 0, 0, 0], dtype=np.float64)
    check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
    cub = pyomo_nlp.constraints_ub()
    comparison_cub = np.asarray([0, 0, 0, 0], dtype=np.float64)
    check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)
    xinit = pyomo_nlp.init_primals()
    comparison_xinit = np.asarray([100, 2, 3, 80, 70, 75, 50], dtype=np.float64)
    check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
    duals_init = pyomo_nlp.init_duals()
    comparison_duals_init = np.asarray([0, 0, 0, 0], dtype=np.float64)
    check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)
    self.assertEqual(7, len(pyomo_nlp.create_new_vector('primals')))
    self.assertEqual(4, len(pyomo_nlp.create_new_vector('constraints')))
    self.assertEqual(4, len(pyomo_nlp.create_new_vector('duals')))
    pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.float64))
    x = pyomo_nlp.get_primals()
    self.assertTrue(np.array_equal(x, np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.float64)))
    pyomo_nlp.set_primals(pyomo_nlp.init_primals())
    pyomo_nlp.set_duals(np.asarray([42, 10, 11, 12], dtype=np.float64))
    y = pyomo_nlp.get_duals()
    self.assertTrue(np.array_equal(y, np.asarray([42, 10, 11, 12], dtype=np.float64)))
    pyomo_nlp.set_duals(np.asarray([21, 5, 6, 7], dtype=np.float64))
    y = pyomo_nlp.get_duals()
    self.assertTrue(np.array_equal(y, np.asarray([21, 5, 6, 7], dtype=np.float64)))
    fac = pyomo_nlp.get_obj_factor()
    self.assertEqual(fac, 1)
    pyomo_nlp.set_obj_factor(42)
    self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
    pyomo_nlp.set_obj_factor(1)
    f = pyomo_nlp.evaluate_objective()
    self.assertEqual(f, 900)
    gradf = pyomo_nlp.evaluate_grad_objective()
    comparison_gradf = np.asarray([0, 0, 0, 0, 0, 0, 60], dtype=np.float64)
    check_vectors_specific_order(self, gradf, x_order, comparison_gradf, comparison_x_order)
    c = pyomo_nlp.evaluate_constraints()
    comparison_c = np.asarray([-2, 26, -13, -22], dtype=np.float64)
    check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
    c = np.zeros(4)
    pyomo_nlp.evaluate_constraints(out=c)
    check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
    j = pyomo_nlp.evaluate_jacobian()
    comparison_j = np.asarray([[-1, 9, 12, 1, 0, 0, 0], [0, 18, 24, -1, 1, 0, 0], [0, -9, -12, 1, 0, -1, 0], [1, -36, -48, 0, 0, 0, -1]])
    check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)
    j = 2.0 * j
    pyomo_nlp.evaluate_jacobian(out=j)
    check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)
    if hessian_support:
        h = pyomo_nlp.evaluate_hessian_lag()
        self.assertTrue(h.shape == (7, 7))
        comparison_h = np.asarray([[0, 0, 0, 0, 0, 0, 0], [0, 0, 2 * 3 * 21 + 4 * 3 * 5 + -2 * 3 * 6 + -8 * 3 * 7, 0, 0, 0, 0], [0, 2 * 3 * 21 + 4 * 3 * 5 + -2 * 3 * 6 + -8 * 3 * 7, 2 * 2 * 21 + 4 * 2 * 5 + -2 * 2 * 6 + -8 * 2 * 7, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2 * 1]], dtype=np.float64)
        check_sparse_matrix_specific_order(self, h, x_order, x_order, comparison_h, comparison_x_order, comparison_x_order)
    else:
        with self.assertRaises(NotImplementedError):
            h = pyomo_nlp.evaluate_hessian_lag()