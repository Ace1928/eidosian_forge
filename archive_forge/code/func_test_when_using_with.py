import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_when_using_with(self):
    """ Check normal use cases as a context manager.
        """
    test_object = self.test_object
    with self.assertTraitDoesNotChange(test_object, 'number') as result:
        test_object.flag = True
        test_object.number = 2.0
    msg = 'The assertion result is not None: {0}'.format(result.event)
    self.assertIsNone(result.event, msg=msg)
    with self.assertTraitChanges(test_object, 'number') as result:
        test_object.flag = False
        test_object.number = 5.0
    expected = (test_object, 'number', 2.0, 5.0)
    self.assertSequenceEqual(expected, result.event)
    with self.assertTraitChanges(test_object, 'number', count=2) as result:
        test_object.flag = False
        test_object.number = 4.0
        test_object.number = 3.0
    expected = [(test_object, 'number', 5.0, 4.0), (test_object, 'number', 4.0, 3.0)]
    self.assertSequenceEqual(expected, result.events)
    self.assertSequenceEqual(expected[-1], result.event)
    with self.assertTraitChanges(test_object, 'number') as result:
        test_object.flag = True
        test_object.add_to_number(10.0)
    expected = (test_object, 'number', 3.0, 13.0)
    self.assertSequenceEqual(expected, result.event)
    with self.assertTraitChanges(test_object, 'number', count=3) as result:
        test_object.flag = True
        test_object.add_to_number(10.0)
        test_object.add_to_number(10.0)
        test_object.add_to_number(10.0)
    expected = [(test_object, 'number', 13.0, 23.0), (test_object, 'number', 23.0, 33.0), (test_object, 'number', 33.0, 43.0)]
    self.assertSequenceEqual(expected, result.events)
    self.assertSequenceEqual(expected[-1], result.event)