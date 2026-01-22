import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
class TestClamping(test.TestCase):

    def test_simple_clamp(self):
        result = misc.clamp(1.0, 2.0, 3.0)
        self.assertEqual(2.0, result)
        result = misc.clamp(4.0, 2.0, 3.0)
        self.assertEqual(3.0, result)
        result = misc.clamp(3.0, 4.0, 4.0)
        self.assertEqual(4.0, result)

    def test_invalid_clamp(self):
        self.assertRaises(ValueError, misc.clamp, 0.0, 2.0, 1.0)

    def test_clamped_callback(self):
        calls = []

        def on_clamped():
            calls.append(True)
        misc.clamp(-1, 0.0, 1.0, on_clamped=on_clamped)
        self.assertEqual(1, len(calls))
        calls.pop()
        misc.clamp(0.0, 0.0, 1.0, on_clamped=on_clamped)
        self.assertEqual(0, len(calls))
        misc.clamp(2, 0.0, 1.0, on_clamped=on_clamped)
        self.assertEqual(1, len(calls))