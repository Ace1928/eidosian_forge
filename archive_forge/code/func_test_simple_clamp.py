import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_simple_clamp(self):
    result = misc.clamp(1.0, 2.0, 3.0)
    self.assertEqual(2.0, result)
    result = misc.clamp(4.0, 2.0, 3.0)
    self.assertEqual(3.0, result)
    result = misc.clamp(3.0, 4.0, 4.0)
    self.assertEqual(4.0, result)