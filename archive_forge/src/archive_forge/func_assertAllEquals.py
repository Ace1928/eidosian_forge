import collections
import doctest
import types
from typing import Any, Iterator, Mapping
import unittest
from absl.testing import parameterized
import attr
import numpy as np
import tree
import wrapt
def assertAllEquals(self, a, b):
    self.assertTrue((np.asarray(a) == b).all())