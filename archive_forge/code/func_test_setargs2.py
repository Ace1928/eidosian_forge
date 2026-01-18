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
def test_setargs2(self):
    a = Set()
    b = Set(a)
    with self.assertRaisesRegex(ValueError, 'Error retrieving component IndexedSet\\[None\\]: The component has not been constructed.'):
        c = Set(within=b, dimen=2)
        c.construct()
    a = Set()
    b = Set()
    c = Set(within=b, dimen=1)
    c.construct()
    self.assertEqual(c.domain, b)