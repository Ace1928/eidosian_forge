import pyomo.common.unittest as unittest
import pytest
import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import assert_disjoint_intervals, IntervalData
class TestAssertDisjoint(unittest.TestCase):

    def test_disjoint(self):
        intervals = [(0, 1), (1, 2)]
        assert_disjoint_intervals(intervals)
        intervals = [(2, 3), (0, 1)]
        assert_disjoint_intervals(intervals)
        intervals = [(0, 1), (1, 1)]
        assert_disjoint_intervals(intervals)

    def test_backwards_endpoints(self):
        intervals = [(0, 1), (3, 2)]
        msg = 'Lower endpoint of interval is higher'
        with self.assertRaisesRegex(RuntimeError, msg):
            assert_disjoint_intervals(intervals)

    def test_not_disjoint(self):
        intervals = [(0, 2), (1, 3)]
        msg = 'are not disjoint'
        with self.assertRaisesRegex(RuntimeError, msg):
            assert_disjoint_intervals(intervals)