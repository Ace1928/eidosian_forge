import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_subtrahend_has_extra_elements(self):
    result = misc.sequence_minus([1, 2, 3, 4], [2, 3, 5, 7, 13])
    self.assertEqual([1, 4], result)