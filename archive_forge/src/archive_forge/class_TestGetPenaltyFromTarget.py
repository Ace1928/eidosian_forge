import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
class TestGetPenaltyFromTarget(unittest.TestCase):

    def _make_model(self, n_time_points=3):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=list(range(n_time_points)))
        m.comp = pyo.Set(initialize=['A', 'B'])
        m.var = pyo.Var(m.time, m.comp, initialize={(i, j): 1.1 * i for i, j in m.time * m.comp})
        return m

    def test_constant_setpoint(self):
        m = self._make_model()
        setpoint = {m.var[:, 'A']: 0.3, m.var[:, 'B']: 0.4}
        variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
        m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)
        pred_expr = {(i, t): (m.var[t, 'B'] - 0.4) ** 2 if i == 0 else (m.var[t, 'A'] - 0.3) ** 2 for i, t in m.var_set * m.time}
        for t in m.time:
            for i in m.var_set:
                self.assertTrue(compare_expressions(pred_expr[i, t], m.penalty[i, t].expr))
                self.assertEqual(pyo.value(pred_expr[i, t]), pyo.value(m.penalty[i, t]))

    def test_constant_setpoint_with_ScalarData(self):
        m = self._make_model()
        setpoint = ScalarData({m.var[:, 'A']: 0.3, m.var[:, 'B']: 0.4})
        variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
        m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)
        pred_expr = {(i, t): (m.var[t, 'B'] - 0.4) ** 2 if i == 0 else (m.var[t, 'A'] - 0.3) ** 2 for i, t in m.var_set * m.time}
        for t in m.time:
            for i in m.var_set:
                self.assertTrue(compare_expressions(pred_expr[i, t], m.penalty[i, t].expr))
                self.assertEqual(pyo.value(pred_expr[i, t]), pyo.value(m.penalty[i, t]))

    def test_varying_setpoint(self):
        m = self._make_model(n_time_points=5)
        A_target = [0.4, 0.6, 0.1, 0.0, 1.1]
        B_target = [0.8, 0.9, 1.3, 1.5, 1.4]
        setpoint = ({m.var[:, 'A']: A_target, m.var[:, 'B']: B_target}, m.time)
        variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
        m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)
        target = {(i, t): A_target[j] if i == 1 else B_target[t] for i in m.var_set for j, t in enumerate(m.time)}
        for i, t in m.var_set * m.time:
            pred_expr = (variables[i][t] - target[i, t]) ** 2
            self.assertTrue(compare_expressions(pred_expr, m.penalty[i, t].expr))
            self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i, t]))

    def test_piecewise_constant_setpoint(self):
        m = self._make_model(n_time_points=5)
        A_target = [0.3, 0.9, 0.7]
        B_target = [1.1, 0.1, 0.5]
        setpoint = ({m.var[:, 'A']: A_target, m.var[:, 'B']: B_target}, [(0.0, 0.0), (0.0, 2.0), (2.0, 4.0)])
        variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
        m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)
        target = {(i, j): A_target[j] if i == 1 else B_target[j] for i in m.var_set for j in range(len(A_target))}
        for i, t in m.var_set * m.time:
            if t == 0:
                idx = 0
            elif t <= 2.0:
                idx = 1
            elif t <= 4.0:
                idx = 2
            pred_expr = (variables[i][t] - target[i, idx]) ** 2
            self.assertTrue(compare_expressions(pred_expr, m.penalty[i, t].expr))
            self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i, t]))

    def test_bad_argument(self):
        m = self._make_model(n_time_points=3)
        A_target = [0.4, 0.6, 0.1]
        B_target = [0.8, 0.9, 1.3]
        setpoint = ({m.var[:, 'A']: A_target, m.var[:, 'B']: B_target}, m.time)
        variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
        msg = 'tolerance.*can only be used'
        with self.assertRaisesRegex(RuntimeError, msg):
            m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint, tolerance=1e-08)

    def test_bad_data_tuple(self):
        m = self._make_model(n_time_points=3)
        A_target = [0.4, 0.6, 0.1]
        B_target = [0.8, 0.9, 1.3]
        setpoint = ({m.var[:, 'A']: A_target, m.var[:, 'B']: B_target}, m.time, 'something else')
        variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
        msg = 'tuple of length two'
        with self.assertRaisesRegex(TypeError, msg):
            m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)

    def test_bad_data_tuple_entry_0(self):
        m = self._make_model(n_time_points=3)
        A_target = [0.4, 0.6, 0.1]
        B_target = [0.8, 0.9, 1.3]
        setpoint = ([(m.var[:, 'A'], A_target), (m.var[:, 'B'], B_target)], m.time)
        variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
        msg = 'must be instance of MutableMapping'
        with self.assertRaisesRegex(TypeError, msg):
            m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)

    def test_empty_time_list(self):
        m = self._make_model(n_time_points=3)
        A_target = []
        B_target = []
        setpoint = ({m.var[:, 'A']: A_target, m.var[:, 'B']: B_target}, [])
        variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
        msg = 'Time sequence.*is empty'
        with self.assertRaisesRegex(ValueError, msg):
            m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)

    def test_bad_time_list(self):
        m = self._make_model(n_time_points=3)
        A_target = [0.4, 0.6, 0.1]
        B_target = [0.8, 0.9, 1.3]
        setpoint = (dict([(m.var[:, 'A'], A_target), (m.var[:, 'B'], B_target)]), [0.0, (0.1, 0.2), 0.3])
        variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
        msg = 'Second entry of data tuple must be'
        with self.assertRaisesRegex(TypeError, msg):
            m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)