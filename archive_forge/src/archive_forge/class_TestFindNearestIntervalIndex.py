import pyomo.common.unittest as unittest
import pytest
from pyomo.contrib.mpc.data.find_nearest_index import (
class TestFindNearestIntervalIndex(unittest.TestCase):

    def test_find_interval(self):
        intervals = [(0.0, 0.1), (0.1, 0.5), (0.7, 1.0)]
        target = 0.05
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 0)
        target = 0.099
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 0)
        target = 0.1
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 0)
        target = 0.1
        idx = find_nearest_interval_index(intervals, target, prefer_left=False)
        self.assertEqual(idx, 1)
        target = 0.55
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 1)
        target = 0.6
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 1)
        target = 0.6999
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 2)
        target = 1.0
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 2)
        target = -0.1
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 0)
        target = 1.1
        idx = find_nearest_interval_index(intervals, target)
        self.assertEqual(idx, 2)

    def test_find_interval_tolerance(self):
        intervals = [(0.0, 0.1), (0.1, 0.5), (0.7, 1.0)]
        target = 0.501
        idx = find_nearest_interval_index(intervals, target, tolerance=None)
        self.assertEqual(idx, 1)
        idx = find_nearest_interval_index(intervals, target, tolerance=1e-05)
        self.assertEqual(idx, None)
        idx = find_nearest_interval_index(intervals, target, tolerance=0.01)
        self.assertEqual(idx, 1)
        target = 1.001
        idx = find_nearest_interval_index(intervals, target, tolerance=0.01)
        self.assertEqual(idx, 2)
        idx = find_nearest_interval_index(intervals, target, tolerance=0.0001)
        self.assertEqual(idx, None)

    def test_find_interval_with_tolerance_on_boundary(self):
        intervals = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0)]
        target = 0.1001
        idx = find_nearest_interval_index(intervals, target, tolerance=None, prefer_left=True)
        self.assertEqual(idx, 1)
        idx = find_nearest_interval_index(intervals, target, tolerance=1e-05, prefer_left=True)
        self.assertEqual(idx, 1)
        idx = find_nearest_interval_index(intervals, target, tolerance=1e-05, prefer_left=False)
        self.assertEqual(idx, 1)
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=True)
        self.assertEqual(idx, 0)
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=False)
        self.assertEqual(idx, 1)
        target = 0.4999
        idx = find_nearest_interval_index(intervals, target, tolerance=1e-05, prefer_left=True)
        self.assertEqual(idx, 1)
        idx = find_nearest_interval_index(intervals, target, tolerance=1e-05, prefer_left=False)
        self.assertEqual(idx, 1)
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=True)
        self.assertEqual(idx, 1)
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=False)
        self.assertEqual(idx, 2)

    def test_find_interval_with_tolerance_singleton(self):
        intervals = [(0.0, 0.1), (0.1, 0.1), (0.5, 0.5), (0.5, 1.0)]
        target = 0.1001
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=True)
        self.assertEqual(idx, 0)
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=False)
        self.assertEqual(idx, 1)
        target = 0.0999
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=True)
        self.assertEqual(idx, 0)
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=False)
        self.assertEqual(idx, 1)
        target = 0.4999
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=True)
        self.assertEqual(idx, 2)
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=False)
        self.assertEqual(idx, 3)
        target = 0.5001
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=True)
        self.assertEqual(idx, 2)
        idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=False)
        self.assertEqual(idx, 3)