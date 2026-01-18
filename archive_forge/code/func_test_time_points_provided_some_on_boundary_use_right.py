import pyomo.common.unittest as unittest
import pytest
import random
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentMap
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import (
def test_time_points_provided_some_on_boundary_use_right(self):
    m = _make_model()
    intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
    data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
    interval_data = IntervalData(data, intervals)
    time_points = [0.1 * i for i in range(11)]
    series_data = interval_to_series(interval_data, time_points=time_points, prefer_left=False)
    pred_data = {m.var[:, 'A']: [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], m.var[:, 'B']: [4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]}
    self.assertEqual(series_data, TimeSeriesData(pred_data, time_points))