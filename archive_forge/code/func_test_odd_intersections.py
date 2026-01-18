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
def test_odd_intersections(self):
    m = AbstractModel()
    m.p = Param(initialize=0)
    m.a = RangeSet(0, None, 2)
    m.b = RangeSet(5, 10, m.p, finite=False)
    m.x = m.a & m.b
    self.assertTrue(m.a._constructed)
    self.assertFalse(m.b._constructed)
    self.assertFalse(m.x._constructed)
    self.assertIs(type(m.x), SetIntersection_InfiniteSet)
    i = m.create_instance()
    self.assertIs(type(i.x), SetIntersection_OrderedSet)
    self.assertEqual(list(i.x), [6, 8, 10])
    self.assertEqual(i.x.ord(6), 1)
    self.assertEqual(i.x.ord(8), 2)
    self.assertEqual(i.x.ord(10), 3)
    self.assertEqual(i.x[1], 6)
    self.assertEqual(i.x[2], 8)
    self.assertEqual(i.x[3], 10)
    with self.assertRaisesRegex(IndexError, 'x index out of range'):
        i.x[4]
    self.assertEqual(i.x[-3], 6)
    self.assertEqual(i.x[-2], 8)
    self.assertEqual(i.x[-1], 10)
    with self.assertRaisesRegex(IndexError, 'x index out of range'):
        i.x[-4]