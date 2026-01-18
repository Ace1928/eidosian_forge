import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_piecewise_penalty_exceptions(self):
    m = self._make_model(n_time_points=5)
    variables = [pyo.Reference(m.var[:, 'A']), pyo.Reference(m.var[:, 'B'])]
    setpoint_data = IntervalData({m.var[:, 'A']: [2.0, 2.5]}, [(0, 2), (2, 4)])
    weight_data = {pyo.ComponentUID(m.var[:, 'A']): 10.0, pyo.ComponentUID(m.var[:, 'B']): 0.1}
    msg = 'Setpoint data does not contain'
    with self.assertRaisesRegex(KeyError, msg):
        tr_cost = get_penalty_from_piecewise_constant_target(variables, m.time, setpoint_data, weight_data=weight_data)
    setpoint_data = IntervalData({m.var[:, 'A']: [2.0, 2.5], m.var[:, 'B']: [3.0, 3.5]}, [(0, 2), (2, 4)])
    weight_data = {pyo.ComponentUID(m.var[:, 'A']): 10.0}
    msg = 'Tracking weight does not contain'
    with self.assertRaisesRegex(KeyError, msg):
        tr_cost = get_penalty_from_piecewise_constant_target(variables, m.time, setpoint_data, weight_data=weight_data)