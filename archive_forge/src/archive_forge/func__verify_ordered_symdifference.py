import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
def _verify_ordered_symdifference(self, a, b):
    if isinstance(a, (Set, SetOf, RangeSet)):
        a_ordered = a.isordered()
    else:
        a_ordered = type(a) is list
    if isinstance(b, (Set, SetOf, RangeSet)):
        b_ordered = b.isordered()
    else:
        b_ordered = type(b) is list
    self.assertTrue(a_ordered)
    x = a ^ b
    self.assertIs(type(x), SetSymmetricDifference_OrderedSet)
    self.assertTrue(x.isfinite())
    self.assertTrue(x.isordered())
    self.assertEqual(len(x), 4)
    self.assertEqual(list(x), [3, 2, 5, 0])
    self.assertEqual(x.ordered_data(), (3, 2, 5, 0))
    self.assertEqual(x.sorted_data(), (0, 2, 3, 5))
    self.assertIn(0, x)
    self.assertNotIn(1, x)
    self.assertIn(2, x)
    self.assertIn(3, x)
    self.assertNotIn(4, x)
    self.assertIn(5, x)
    self.assertNotIn(6, x)
    self.assertEqual(x.ord(0), 4)
    self.assertEqual(x.ord(2), 2)
    self.assertEqual(x.ord(3), 1)
    self.assertEqual(x.ord(5), 3)
    with self.assertRaisesRegex(IndexError, 'Cannot identify position of 6 in Set SetSymmetricDifference_OrderedSet'):
        x.ord(6)
    self.assertEqual(x[1], 3)
    self.assertEqual(x[2], 2)
    self.assertEqual(x[3], 5)
    self.assertEqual(x[4], 0)
    with self.assertRaisesRegex(IndexError, 'SetSymmetricDifference_OrderedSet index out of range'):
        x[5]
    self.assertEqual(x[-1], 0)
    self.assertEqual(x[-2], 5)
    self.assertEqual(x[-3], 2)
    self.assertEqual(x[-4], 3)
    with self.assertRaisesRegex(IndexError, 'SetSymmetricDifference_OrderedSet index out of range'):
        x[-5]