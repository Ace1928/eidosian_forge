import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_special_case_for_count(self):
    """ Count equal to 0 should be valid but it is discouraged.
        """
    test_object = TestObject(number=16.0)
    with self.assertTraitChanges(test_object, 'number', count=0):
        test_object.flag = True