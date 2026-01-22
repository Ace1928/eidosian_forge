import pyomo.common.unittest as unittest
import pytest
import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import assert_disjoint_intervals, IntervalData
class TestLoadInputs(unittest.TestCase):

    def make_model(self):
        m = pyo.ConcreteModel()
        m.time = dae.ContinuousSet(initialize=[0, 1, 2, 3, 4, 5, 6])
        m.v = pyo.Var(m.time, initialize=0)
        return m

    def test_load_inputs_some_time(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({'v': [1.0]}, [(2, 4)])
        interface.load_data(inputs)
        for t in m.time:
            if t == 3 or t == 4:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 0.0)

    def test_load_inputs_some_time_include_endpoints(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({'v': [1.0]}, [(2, 4)])
        interface.load_data(inputs, exclude_left_endpoint=False)
        for t in m.time:
            if t == 2 or t == 3 or t == 4:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 0.0)

    def test_load_inputs_some_time_exclude_endpoints(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({'v': [1.0]}, [(2, 4)])
        interface.load_data(inputs, exclude_right_endpoint=True)
        for t in m.time:
            if t == 3:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 0.0)

    def test_load_inputs_all_time_default(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({'v': [1.0, 2.0]}, [(0, 3), (3, 6)])
        interface.load_data(inputs)
        for t in m.time:
            if t == 0:
                self.assertEqual(m.v[t].value, 0.0)
            elif t <= 3:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 2.0)

    def test_load_inputs_all_time_prefer_right(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({'v': [1.0, 2.0]}, [(0, 3), (3, 6)])
        interface.load_data(inputs, prefer_left=False)
        for t in m.time:
            if t < 3:
                self.assertEqual(m.v[t].value, 1.0)
            elif t == 6:
                self.assertEqual(m.v[t].value, 0.0)
            else:
                self.assertEqual(m.v[t].value, 2.0)

    def test_load_inputs_all_time_prefer_right_dont_exclude(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({'v': [1.0, 2.0]}, [(0, 3), (3, 6)])
        interface.load_data(inputs, prefer_left=False, exclude_right_endpoint=False)
        for t in m.time:
            if t < 3:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 2.0)

    def load_inputs_invalid_time(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({'v': [1.0, 2.0, 3.0]}, [(0, 3), (3, 6), (6, 9)])
        interface.load_data(inputs)
        for t in m.time:
            if t == 0:
                self.assertEqual(m.v[t].value, 0.0)
            elif t <= 3:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 2.0)

    def load_inputs_exception(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = {'_v': {(0, 3): 1.0, (3, 6): 2.0, (6, 9): 3.0}}
        inputs = mpc.IntervalData({'_v': [1.0, 2.0, 3.0]}, [(0, 3), (3, 6), (6, 9)])
        with self.assertRaisesRegex(RuntimeError, 'Cannot find'):
            interface.load_data(inputs)