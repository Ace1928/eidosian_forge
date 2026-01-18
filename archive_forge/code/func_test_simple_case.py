import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_simple_case(self):
    result = misc.sequence_minus([1, 2, 3, 4], [2, 3])
    self.assertEqual([1, 4], result)