import unittest
from numba import njit
from numba.tests.support import TestCase, override_config
from numba.misc import llvm_pass_timings as lpt
def test_parse_raw(self):
    timings1 = lpt.ProcessedPassTimings(timings_raw1)
    self.assertAlmostEqual(timings1.get_total_time(), 0.0001)
    self.assertIsInstance(timings1.summary(), str)
    timings2 = lpt.ProcessedPassTimings(timings_raw2)
    self.assertAlmostEqual(timings2.get_total_time(), 0.0001)
    self.assertIsInstance(timings2.summary(), str)