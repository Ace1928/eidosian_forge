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
def _verify_ordered_union(self, a, b):
    if isinstance(a, SetOf):
        self.assertTrue(a.isordered())
        self.assertTrue(a.isfinite())
    else:
        self.assertIs(type(a), list)
    if isinstance(b, SetOf):
        self.assertTrue(b.isordered())
        self.assertTrue(b.isfinite())
    else:
        self.assertIs(type(b), list)
    x = a | b
    self.assertIs(type(x), SetUnion_OrderedSet)
    self.assertTrue(x.isfinite())
    self.assertTrue(x.isordered())
    self.assertEqual(len(x), 5)
    self.assertEqual(list(x), [1, 3, 2, 5, 4])
    self.assertEqual(x.ordered_data(), (1, 3, 2, 5, 4))
    self.assertEqual(x.sorted_data(), (1, 2, 3, 4, 5))
    self.assertIn(1, x)
    self.assertIn(2, x)
    self.assertIn(3, x)
    self.assertIn(4, x)
    self.assertIn(5, x)
    self.assertNotIn(6, x)
    self.assertEqual(x.ord(1), 1)
    self.assertEqual(x.ord(2), 3)
    self.assertEqual(x.ord(3), 2)
    self.assertEqual(x.ord(4), 5)
    self.assertEqual(x.ord(5), 4)
    with self.assertRaisesRegex(IndexError, 'Cannot identify position of 6 in Set SetUnion_OrderedSet'):
        x.ord(6)
    self.assertEqual(x[1], 1)
    self.assertEqual(x[2], 3)
    self.assertEqual(x[3], 2)
    self.assertEqual(x[4], 5)
    self.assertEqual(x[5], 4)
    with self.assertRaisesRegex(IndexError, 'SetUnion_OrderedSet index out of range'):
        x[6]
    self.assertEqual(x[-1], 4)
    self.assertEqual(x[-2], 5)
    self.assertEqual(x[-3], 2)
    self.assertEqual(x[-4], 3)
    self.assertEqual(x[-5], 1)
    with self.assertRaisesRegex(IndexError, 'SetUnion_OrderedSet index out of range'):
        x[-6]