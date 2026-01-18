import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
@unittest.skipIf(not _has_numpy, 'Numpy is not installed')
def test_numpy_membership(self):
    self.assertEqual(numpy.int_(0) in Boolean, True)
    self.assertEqual(numpy.int_(1) in Boolean, True)
    self.assertEqual(numpy.bool_(True) in Boolean, True)
    self.assertEqual(numpy.bool_(False) in Boolean, True)
    self.assertEqual(numpy.float_(1.1) in Boolean, False)
    self.assertEqual(numpy.int_(2) in Boolean, False)
    self.assertEqual(numpy.int_(0) in Integers, True)
    self.assertEqual(numpy.int_(1) in Integers, True)
    self.assertEqual(numpy.bool_(True) in Integers, True)
    self.assertEqual(numpy.bool_(False) in Integers, True)
    self.assertEqual(numpy.float_(1.1) in Integers, False)
    self.assertEqual(numpy.int_(2) in Integers, True)
    self.assertEqual(numpy.int_(0) in Reals, True)
    self.assertEqual(numpy.int_(1) in Reals, True)
    self.assertEqual(numpy.bool_(True) in Reals, True)
    self.assertEqual(numpy.bool_(False) in Reals, True)
    self.assertEqual(numpy.float_(1.1) in Reals, True)
    self.assertEqual(numpy.int_(2) in Reals, True)
    self.assertEqual(numpy.int_(0) in Any, True)
    self.assertEqual(numpy.int_(1) in Any, True)
    self.assertEqual(numpy.bool_(True) in Any, True)
    self.assertEqual(numpy.bool_(False) in Any, True)
    self.assertEqual(numpy.float_(1.1) in Any, True)
    self.assertEqual(numpy.int_(2) in Any, True)