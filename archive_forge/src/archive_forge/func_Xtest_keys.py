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
def Xtest_keys(self):
    """Check the keys for the array"""
    tmp = self.instance.A.keys()
    tmp.sort()
    self.assertEqual(tmp, ['A', 'C'])