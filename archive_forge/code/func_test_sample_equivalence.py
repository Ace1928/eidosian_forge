import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_sample_equivalence(self):
    expected = list(reversed(list(enumerate(self.sample))))
    actual = list(misc.reverse_enumerate(self.sample))
    self.assertEqual(expected, actual)