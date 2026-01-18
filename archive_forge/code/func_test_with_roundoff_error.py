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
def test_with_roundoff_error(self):
    m = _make_model()
    intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
    data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
    interval_data = IntervalData(data, intervals)
    random.seed(12710)
    time_points = [i * 0.1 + random.uniform(-1e-08, 1e-08) for i in range(11)]
    series_data = interval_to_series(interval_data, time_points=time_points, tolerance=1e-07)
    pred_data = {m.var[:, 'A']: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0], m.var[:, 'B']: [4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]}
    self.assertEqual(series_data, TimeSeriesData(pred_data, time_points))