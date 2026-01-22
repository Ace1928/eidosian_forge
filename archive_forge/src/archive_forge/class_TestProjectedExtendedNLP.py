import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
class TestProjectedExtendedNLP(unittest.TestCase):

    def _make_model_with_inequalities(self):
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=range(4))
        m.x = pyo.Var(m.I, initialize=1.1)
        m.obj = pyo.Objective(expr=1 * m.x[0] + 2 * m.x[1] ** 2 + 3 * m.x[1] * m.x[2] + 4 * m.x[3] ** 3)
        m.eq_con_1 = pyo.Constraint(expr=m.x[0] * m.x[1] ** 1.1 * m.x[2] ** 1.2 == 3.0)
        m.eq_con_2 = pyo.Constraint(expr=m.x[0] ** 2 + m.x[3] ** 2 + m.x[1] == 2.0)
        m.ineq_con_1 = pyo.Constraint(expr=m.x[0] + m.x[3] * m.x[0] <= 4.0)
        m.ineq_con_2 = pyo.Constraint(expr=m.x[1] + m.x[2] >= 1.0)
        m.ineq_con_3 = pyo.Constraint(expr=m.x[2] >= 0)
        return m

    def _get_nlps(self):
        m = self._make_model_with_inequalities()
        nlp = PyomoNLP(m)
        primals_ordering = ['x[1]', 'x[0]']
        proj_nlp = ProjectedExtendedNLP(nlp, primals_ordering)
        return (m, nlp, proj_nlp)

    def _x_to_nlp(self, m, nlp, values):
        indices = nlp.get_primal_indices([m.x[0], m.x[1], m.x[2], m.x[3]])
        reordered_values = [None for _ in m.x]
        for i, val in zip(indices, values):
            reordered_values[i] = val
        return reordered_values

    def _c_to_nlp(self, m, nlp, values):
        indices = nlp.get_constraint_indices([m.eq_con_1, m.eq_con_2, m.ineq_con_1, m.ineq_con_2, m.ineq_con_3])
        reordered_values = [None] * 5
        for i, val in zip(indices, values):
            reordered_values[i] = val
        return reordered_values

    def _eq_to_nlp(self, m, nlp, values):
        indices = nlp.get_equality_constraint_indices([m.eq_con_1, m.eq_con_2])
        reordered_values = [None] * 2
        for i, val in zip(indices, values):
            reordered_values[i] = val
        return reordered_values

    def _ineq_to_nlp(self, m, nlp, values):
        indices = nlp.get_inequality_constraint_indices([m.ineq_con_1, m.ineq_con_2, m.ineq_con_3])
        reordered_values = [None] * 3
        for i, val in zip(indices, values):
            reordered_values[i] = val
        return reordered_values

    def _rc_to_nlp(self, m, nlp, rc):
        var_indices = nlp.get_primal_indices(list(m.x.values()))
        con_indices = nlp.get_constraint_indices([m.eq_con_1, m.eq_con_2, m.ineq_con_1, m.ineq_con_2, m.ineq_con_3])
        i, j = rc
        return (con_indices[i], var_indices[j])

    def _rc_to_proj_nlp(self, m, nlp, rc):
        var_indices = [1, 0]
        con_indices = nlp.get_constraint_indices([m.eq_con_1, m.eq_con_2, m.ineq_con_1, m.ineq_con_2, m.ineq_con_3])
        i, j = rc
        return (con_indices[i], var_indices[j])

    def _rc_to_proj_nlp_eq(self, m, nlp, rc):
        var_indices = [1, 0]
        con_indices = nlp.get_equality_constraint_indices([m.eq_con_1, m.eq_con_2])
        i, j = rc
        return (con_indices[i], var_indices[j])

    def _rc_to_proj_nlp_ineq(self, m, nlp, rc):
        var_indices = [1, 0]
        con_indices = nlp.get_inequality_constraint_indices([m.ineq_con_1, m.ineq_con_2, m.ineq_con_3])
        i, j = rc
        return (con_indices[i], var_indices[j])

    def test_non_extended_original_nlp(self):
        m, nlp, proj_nlp = self._get_nlps()
        proj_nlp = ProjectedNLP(nlp, ['x[0]', 'x[1]', 'x[2]'])
        msg = 'Original NLP must be an instance of ExtendedNLP'
        with self.assertRaisesRegex(TypeError, msg):
            proj_ext_nlp = ProjectedExtendedNLP(proj_nlp, ['x[1]', 'x[0]'])

    def test_n_primals_constraints(self):
        m, nlp, proj_nlp = self._get_nlps()
        self.assertEqual(proj_nlp.n_primals(), 2)
        self.assertEqual(proj_nlp.n_constraints(), 5)
        self.assertEqual(proj_nlp.n_eq_constraints(), 2)
        self.assertEqual(proj_nlp.n_ineq_constraints(), 3)

    def test_set_get_primals(self):
        m, nlp, proj_nlp = self._get_nlps()
        primals = proj_nlp.get_primals()
        np.testing.assert_array_equal(primals, [1.1, 1.1])
        nlp.set_primals(self._x_to_nlp(m, nlp, [1.2, 1.3, 1.4, 1.5]))
        proj_primals = proj_nlp.get_primals()
        np.testing.assert_array_equal(primals, [1.3, 1.2])
        proj_nlp.set_primals(np.array([-1.0, -1.1]))
        np.testing.assert_array_equal(proj_nlp.get_primals(), [-1.0, -1.1])
        np.testing.assert_array_equal(nlp.get_primals(), self._x_to_nlp(m, nlp, [-1.1, -1.0, 1.4, 1.5]))

    def test_set_primals_with_list_error(self):
        m, nlp, proj_nlp = self._get_nlps()
        msg = 'only integer scalar arrays can be converted to a scalar index'
        with self.assertRaisesRegex(TypeError, msg):
            proj_nlp.set_primals([1.0, 2.0])

    def test_get_set_duals(self):
        m, nlp, proj_nlp = self._get_nlps()
        nlp.set_duals([2, 3, 4, 5, 6])
        np.testing.assert_array_equal(proj_nlp.get_duals(), [2, 3, 4, 5, 6])
        proj_nlp.set_duals([-1, -2, -3, -4, -5])
        np.testing.assert_array_equal(proj_nlp.get_duals(), [-1, -2, -3, -4, -5])
        np.testing.assert_array_equal(nlp.get_duals(), [-1, -2, -3, -4, -5])

    def test_eval_constraints(self):
        m, nlp, proj_nlp = self._get_nlps()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))
        con_resids = nlp.evaluate_constraints()
        pred_con_body = [x0 * x1 ** 1.1 * x2 ** 1.2 - 3.0, x0 ** 2 + x3 ** 2 + x1 - 2.0, x0 + x0 * x3, x1 + x2, x2]
        np.testing.assert_array_equal(con_resids, self._c_to_nlp(m, nlp, pred_con_body))
        con_resids = proj_nlp.evaluate_constraints()
        np.testing.assert_array_equal(con_resids, self._c_to_nlp(m, nlp, pred_con_body))
        eq_resids = proj_nlp.evaluate_eq_constraints()
        pred_eq_body = [x0 * x1 ** 1.1 * x2 ** 1.2 - 3.0, x0 ** 2 + x3 ** 2 + x1 - 2.0]
        np.testing.assert_array_equal(eq_resids, self._eq_to_nlp(m, nlp, pred_eq_body))
        ineq_body = proj_nlp.evaluate_ineq_constraints()
        pred_ineq_body = [x0 + x0 * x3, x1 + x2, x2]
        np.testing.assert_array_equal(ineq_body, self._ineq_to_nlp(m, nlp, pred_ineq_body))

    def test_eval_jacobian_orig_nlp(self):
        m, nlp, proj_nlp = self._get_nlps()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))
        jac = nlp.evaluate_jacobian()
        pred_rc = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2), (4, 2)]
        pred_data_dict = {(0, 0): x1 ** 1.1 * x2 ** 1.2, (0, 1): 1.1 * x0 * x1 ** 0.1 * x2 ** 1.2, (0, 2): 1.2 * x0 * x1 ** 1.1 * x2 ** 0.2, (1, 0): 2 * x0, (1, 1): 1.0, (1, 3): 2 * x3, (2, 0): 1.0 + x3, (2, 3): x0, (3, 1): 1.0, (3, 2): 1.0, (4, 2): 1.0}
        pred_rc_set = set((self._rc_to_nlp(m, nlp, rc) for rc in pred_rc))
        pred_data_dict = {self._rc_to_nlp(m, nlp, rc): val for rc, val in pred_data_dict.items()}
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)
        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)

    def test_eval_jacobian_proj_nlp(self):
        m, nlp, proj_nlp = self._get_nlps()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))
        jac = proj_nlp.evaluate_jacobian()
        self.assertEqual(jac.shape, (5, 2))
        pred_rc = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 1)]
        pred_data_dict = {(0, 0): x1 ** 1.1 * x2 ** 1.2, (0, 1): 1.1 * x0 * x1 ** 0.1 * x2 ** 1.2, (1, 0): 2 * x0, (1, 1): 1.0, (2, 0): 1.0 + x3, (3, 1): 1.0}
        pred_rc_set = set((self._rc_to_proj_nlp(m, nlp, rc) for rc in pred_rc))
        pred_data_dict = {self._rc_to_proj_nlp(m, nlp, rc): val for rc, val in pred_data_dict.items()}
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)
        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)

    def test_eval_eq_jacobian_proj_nlp(self):
        m, nlp, proj_nlp = self._get_nlps()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))
        jac = proj_nlp.evaluate_jacobian_eq()
        self.assertEqual(jac.shape, (2, 2))
        pred_rc = [(0, 0), (0, 1), (1, 0), (1, 1)]
        pred_data_dict = {(0, 0): x1 ** 1.1 * x2 ** 1.2, (0, 1): 1.1 * x0 * x1 ** 0.1 * x2 ** 1.2, (1, 0): 2 * x0, (1, 1): 1.0}
        pred_rc_set = set((self._rc_to_proj_nlp_eq(m, nlp, rc) for rc in pred_rc))
        pred_data_dict = {self._rc_to_proj_nlp_eq(m, nlp, rc): val for rc, val in pred_data_dict.items()}
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)
        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)

    def test_eval_ineq_jacobian_proj_nlp(self):
        m, nlp, proj_nlp = self._get_nlps()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))
        jac = proj_nlp.evaluate_jacobian_ineq()
        self.assertEqual(jac.shape, (3, 2))
        pred_rc = [(0, 0), (1, 1)]
        pred_data_dict = {(0, 0): 1.0 + x3, (1, 1): 1.0}
        pred_rc_set = set((self._rc_to_proj_nlp_ineq(m, nlp, rc) for rc in pred_rc))
        pred_data_dict = {self._rc_to_proj_nlp_ineq(m, nlp, rc): val for rc, val in pred_data_dict.items()}
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)
        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)

    def test_eval_eq_jacobian_proj_nlp_using_out_arg(self):
        m, nlp, proj_nlp = self._get_nlps()
        jac = proj_nlp.evaluate_jacobian_eq()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))
        proj_nlp.evaluate_jacobian_eq(out=jac)
        self.assertEqual(jac.shape, (2, 2))
        pred_rc = [(0, 0), (0, 1), (1, 0), (1, 1)]
        pred_data_dict = {(0, 0): x1 ** 1.1 * x2 ** 1.2, (0, 1): 1.1 * x0 * x1 ** 0.1 * x2 ** 1.2, (1, 0): 2 * x0, (1, 1): 1.0}
        pred_rc_set = set((self._rc_to_proj_nlp_eq(m, nlp, rc) for rc in pred_rc))
        pred_data_dict = {self._rc_to_proj_nlp_eq(m, nlp, rc): val for rc, val in pred_data_dict.items()}
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)
        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)

    def test_eval_ineq_jacobian_proj_nlp_using_out_arg(self):
        m, nlp, proj_nlp = self._get_nlps()
        jac = proj_nlp.evaluate_jacobian_ineq()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))
        proj_nlp.evaluate_jacobian_ineq(out=jac)
        self.assertEqual(jac.shape, (3, 2))
        pred_rc = [(0, 0), (1, 1)]
        pred_data_dict = {(0, 0): 1.0 + x3, (1, 1): 1.0}
        pred_rc_set = set((self._rc_to_proj_nlp_ineq(m, nlp, rc) for rc in pred_rc))
        pred_data_dict = {self._rc_to_proj_nlp_ineq(m, nlp, rc): val for rc, val in pred_data_dict.items()}
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)
        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)