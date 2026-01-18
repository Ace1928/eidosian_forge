import sys
import unittest
from Cython.Utils import (
def test_requirements_for_Cached(self):
    obj = Cached()
    self.assertFalse(hasattr(obj, CACHE_NAME))
    self.assertTrue(hasattr(obj, METHOD_NAME))
    self.set_of_names_equal(obj, set())