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
def Xtest_eq2(self):
    """Various checks for set equality and inequality (2)"""
    try:
        self.assertEqual(self.instance.A == self.instance.tmpset2, False)
        self.assertEqual(self.instance.tmpset2 == self.instance.A, False)
        self.assertEqual(self.instance.A != self.instance.tmpset2, True)
        self.assertEqual(self.instance.tmpset2 != self.instance.A, True)
    except TypeError:
        pass
    else:
        self.fail('fail test_eq2')