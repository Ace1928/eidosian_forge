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
def test_rangeset_iter(self):
    i = RangeSet(0, 10, 2)
    self.assertEqual(tuple(i), (0, 2, 4, 6, 8, 10))
    self.assertEqual(tuple(i.ordered_iter()), (0, 2, 4, 6, 8, 10))
    self.assertEqual(tuple(i.sorted_iter()), (0, 2, 4, 6, 8, 10))
    i = RangeSet(ranges=(NR(0, 5, 2), NR(6, 10, 2)))
    self.assertEqual(tuple(i), (0, 2, 4, 6, 8, 10))
    i = RangeSet(ranges=(NR(0, 10, 2), NR(0, 10, 2)))
    self.assertEqual(tuple(i), (0, 2, 4, 6, 8, 10))
    i = RangeSet(ranges=(NR(0, 10, 2), NR(10, 0, -2)))
    self.assertEqual(tuple(i), (0, 2, 4, 6, 8, 10))
    i = RangeSet(ranges=(NR(0, 10, 2), NR(9, 0, -2)))
    self.assertEqual(tuple(i), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    i = RangeSet(ranges=(NR(0, 10, 2), NR(1, 10, 2)))
    self.assertEqual(tuple(i), tuple(range(11)))
    i = RangeSet(ranges=(NR(0, 30, 10), NR(12, 14, 1)))
    self.assertEqual(tuple(i), (0, 10, 12, 13, 14, 20, 30))
    i = RangeSet(ranges=(NR(0, 0, 0), NR(3, 3, 0), NR(2, 2, 0)))
    self.assertEqual(tuple(i), (0, 2, 3))