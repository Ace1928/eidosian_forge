import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_piecewise_constant_setpoint_with_specified_variables(self):
    m = self._make_model(n_time_points=5)
    interface = DynamicModelInterface(m, m.time)
    A_target = [0.3, 0.9, 0.7]
    B_target = [1.1, 0.1, 0.5]
    setpoint = ({m.var[:, 'A']: A_target, m.var[:, 'B']: B_target}, [(0.0, 0.0), (0.0, 2.0), (2.0, 4.0)])
    variables = [pyo.Reference(m.var[:, 'B'])]
    m.var_set, m.penalty = interface.get_penalty_from_target(setpoint, variables=variables)
    self.assertEqual(len(m.var_set), 1)
    self.assertEqual(m.var_set[1], 0)
    for i, t in m.var_set * m.time:
        if t == 0:
            idx = 0
        elif t <= 2.0:
            idx = 1
        elif t <= 4.0:
            idx = 2
        var = m.var[t, 'B']
        pred_expr = (var - B_target[idx]) ** 2
        self.assertTrue(compare_expressions(pred_expr, m.penalty[i, t].expr))
        self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i, t]))