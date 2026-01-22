import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def assertCompareEqual(self, k1, k2):
    self.assertTrue(k1 == k2)
    self.assertTrue(k1 <= k2)
    self.assertTrue(k1 >= k2)
    self.assertFalse(k1 != k2)
    self.assertFalse(k1 < k2)
    self.assertFalse(k1 > k2)