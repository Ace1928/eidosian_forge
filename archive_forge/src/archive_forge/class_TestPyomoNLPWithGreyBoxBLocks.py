import itertools
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import (
class TestPyomoNLPWithGreyBoxBLocks(unittest.TestCase):

    def test_set_and_evaluate(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block
        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
        block.set_external_model(ex_model)
        a = m.ex_block.inputs['input_0']
        b = m.ex_block.inputs['input_1']
        r = m.ex_block.inputs['input_2']
        x = m.ex_block.inputs['input_3']
        y = m.ex_block.inputs['input_4']
        m.obj = pyo.Objective(expr=(x - 2.0) ** 2 + (y - 2.0) ** 2 + (a - 2.0) ** 2 + (b - 2.0) ** 2 + (r - 2.0) ** 2)
        _add_linking_constraints(m)
        nlp = PyomoNLPWithGreyBoxBlocks(m)
        self.assertEqual(nlp.n_primals(), 8)
        primals_names = ['a', 'b', 'ex_block.inputs[input_0]', 'ex_block.inputs[input_1]', 'ex_block.inputs[input_2]', 'ex_block.inputs[input_3]', 'ex_block.inputs[input_4]', 'r']
        self.assertEqual(nlp.primals_names(), primals_names)
        np.testing.assert_equal(np.zeros(8), nlp.get_primals())
        primals = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        nlp.set_primals(primals)
        np.testing.assert_equal(primals, nlp.get_primals())
        nlp.load_state_into_pyomo()
        for name, val in zip(primals_names, primals):
            var = m.find_component(name)
            self.assertEqual(var.value, val)
        constraint_names = ['linking_constraint[0]', 'linking_constraint[1]', 'linking_constraint[2]', 'ex_block.residual_0', 'ex_block.residual_1']
        self.assertEqual(constraint_names, nlp.constraint_names())
        residuals = np.array([-2.0, -2.0, 3.0, 5.0 - -3.03051522, 6.0 - 3.583839997])
        np.testing.assert_allclose(residuals, nlp.evaluate_constraints(), rtol=1e-08)
        duals = np.array([1, 2, 3, 4, 5])
        nlp.set_duals(duals)
        self.assertEqual(ex_model.residual_con_multipliers, [4, 5])
        np.testing.assert_equal(nlp.get_duals(), duals)

    def test_jacobian(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block
        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
        block.set_external_model(ex_model)
        a = m.ex_block.inputs['input_0']
        b = m.ex_block.inputs['input_1']
        r = m.ex_block.inputs['input_2']
        x = m.ex_block.inputs['input_3']
        y = m.ex_block.inputs['input_4']
        m.obj = pyo.Objective(expr=(x - 2.0) ** 2 + (y - 2.0) ** 2 + (a - 2.0) ** 2 + (b - 2.0) ** 2 + (r - 2.0) ** 2)
        _add_linking_constraints(m)
        nlp = PyomoNLPWithGreyBoxBlocks(m)
        primals = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        nlp.set_primals(primals)
        jac = nlp.evaluate_jacobian()
        row = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
        col = [0, 2, 1, 3, 7, 4, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6]
        data = [1, -1, 1, -1, 1, -1, -0.16747094, -1.00068434, 1.72383729, 1, 0, -0.30708535, -0.28546127, -0.25235924, 0, 1]
        self.assertEqual(len(row), len(jac.row))
        rcd_dict = dict((((i, j), val) for i, j, val in zip(row, col, data)))
        for i, j, val in zip(jac.row, jac.col, jac.data):
            self.assertIn((i, j), rcd_dict)
            self.assertAlmostEqual(rcd_dict[i, j], val, delta=1e-08)

    def test_hessian_1(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block
        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
        block.set_external_model(ex_model)
        a = m.ex_block.inputs['input_0']
        b = m.ex_block.inputs['input_1']
        r = m.ex_block.inputs['input_2']
        x = m.ex_block.inputs['input_3']
        y = m.ex_block.inputs['input_4']
        m.obj = pyo.Objective(expr=(x - 2.0) ** 2 + (y - 2.0) ** 2 + (a - 2.0) ** 2 + (b - 2.0) ** 2 + (r - 2.0) ** 2)
        _add_nonlinear_linking_constraints(m)
        nlp = PyomoNLPWithGreyBoxBlocks(m)
        primals = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        duals = np.array([1, 1, 1, 1, 1])
        nlp.set_primals(primals)
        nlp.set_duals(duals)
        hess = nlp.evaluate_hessian_lag()
        row = [0, 1, 7]
        col = [0, 1, 7]
        data = [2.0, 2.0, 2.0]
        rcd_dict = dict((((i, j), val) for i, j, val in zip(row, col, data)))
        ex_block_nonzeros = {(2, 2): 2.0 + -1.0 + -0.10967928 + -0.25595929, (2, 3): -0.10684633 + 0.05169308, (3, 2): -0.10684633 + 0.05169308, (2, 4): 0.19329898 + 0.03823075, (4, 2): 0.19329898 + 0.03823075, (3, 3): 2.0 + -1.0 + -1.31592135 + -0.0241836, (3, 4): 1.13920361 + 0.01063667, (4, 3): 1.13920361 + 0.01063667, (4, 4): 2.0 + -1.0 + -1.0891866 + 0.01190218, (5, 5): 2.0, (6, 6): 2.0}
        rcd_dict.update(ex_block_nonzeros)
        ex_block_coords = [2, 3, 4, 5, 6]
        for i, j in itertools.product(ex_block_coords, ex_block_coords):
            row.append(i)
            col.append(j)
            if (i, j) not in rcd_dict:
                rcd_dict[i, j] = 0.0
        self.assertEqual(len(row), len(hess.row))
        for i, j, val in zip(hess.row, hess.col, hess.data):
            self.assertIn((i, j), rcd_dict)
            self.assertAlmostEqual(rcd_dict[i, j], val, delta=1e-08)

    def test_hessian_2(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block
        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
        block.set_external_model(ex_model)
        a = m.ex_block.inputs['input_0']
        b = m.ex_block.inputs['input_1']
        r = m.ex_block.inputs['input_2']
        x = m.ex_block.inputs['input_3']
        y = m.ex_block.inputs['input_4']
        m.obj = pyo.Objective(expr=(x - 2.0) ** 2 + (y - 2.0) ** 2 + (a - 2.0) ** 2 + (b - 2.0) ** 2 + (r - 2.0) ** 2)
        _add_nonlinear_linking_constraints(m)
        nlp = PyomoNLPWithGreyBoxBlocks(m)
        primals = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        duals = np.array([4.4, -3.3, 2.2, -1.1, 0.0])
        nlp.set_primals(primals)
        nlp.set_duals(duals)
        hess = nlp.evaluate_hessian_lag()
        row = [0, 1, 7]
        col = [0, 1, 7]
        data = [4.4 * 2.0, -3.3 * 2.0, 2.2 * 2.0]
        rcd_dict = dict((((i, j), val) for i, j, val in zip(row, col, data)))
        ex_block_nonzeros = {(2, 2): 2.0 + 4.4 * -1.0 + -1.1 * -0.10967928, (2, 3): -1.1 * -0.10684633, (3, 2): -1.1 * -0.10684633, (2, 4): -1.1 * 0.19329898, (4, 2): -1.1 * 0.19329898, (3, 3): 2.0 + -3.3 * -1.0 + -1.1 * -1.31592135, (3, 4): -1.1 * 1.13920361, (4, 3): -1.1 * 1.13920361, (4, 4): 2.0 + 2.2 * -1.0 + -1.1 * -1.0891866, (5, 5): 2.0, (6, 6): 2.0}
        rcd_dict.update(ex_block_nonzeros)
        ex_block_coords = [2, 3, 4, 5, 6]
        for i, j in itertools.product(ex_block_coords, ex_block_coords):
            row.append(i)
            col.append(j)
            if (i, j) not in rcd_dict:
                rcd_dict[i, j] = 0.0
        self.assertEqual(len(row), len(hess.row))
        for i, j, val in zip(hess.row, hess.col, hess.data):
            self.assertIn((i, j), rcd_dict)
            self.assertAlmostEqual(rcd_dict[i, j], val, delta=1e-08)