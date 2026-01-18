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
def test_setargs1(self):
    try:
        a = Set()
        c = Set(a, foo=None)
        self.fail('test_setargs1 - expected error because of bad argument')
    except ValueError:
        pass