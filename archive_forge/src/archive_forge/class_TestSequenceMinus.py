import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
class TestSequenceMinus(test.TestCase):

    def test_simple_case(self):
        result = misc.sequence_minus([1, 2, 3, 4], [2, 3])
        self.assertEqual([1, 4], result)

    def test_subtrahend_has_extra_elements(self):
        result = misc.sequence_minus([1, 2, 3, 4], [2, 3, 5, 7, 13])
        self.assertEqual([1, 4], result)

    def test_some_items_are_equal(self):
        result = misc.sequence_minus([1, 1, 1, 1], [1, 1, 3])
        self.assertEqual([1, 1], result)

    def test_equal_items_not_continious(self):
        result = misc.sequence_minus([1, 2, 3, 1], [1, 3])
        self.assertEqual([2, 1], result)