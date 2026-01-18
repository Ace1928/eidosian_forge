import pyomo.common.unittest as unittest
import pytest
import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import assert_disjoint_intervals, IntervalData
def test_shift_time_points(self):
    m = self._make_model()
    intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
    data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
    interval_data = IntervalData(data, intervals)
    interval_data.shift_time_points(1.0)
    intervals = [(1.0, 1.2), (1.2, 1.5), (1.5, 2.0)]
    data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
    new_interval_data = IntervalData(data, intervals)
    self.assertEqual(interval_data, new_interval_data)