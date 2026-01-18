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
def test_ord_index(self):
    r = RangeSet(2, 10, 2)
    for i, v in enumerate([2, 4, 6, 8, 10]):
        self.assertEqual(r.ord(v), i + 1)
        self.assertEqual(r[i + 1], v)
    with self.assertRaisesRegex(IndexError, 'Accessing Pyomo Sets by position is 1-based: valid Set positional index values are \\[1 .. len\\(Set\\)\\] or \\[-1 .. -len\\(Set\\)\\]'):
        r[0]
    with self.assertRaisesRegex(IndexError, 'FiniteScalarRangeSet index out of range'):
        r[10]
    with self.assertRaisesRegex(ValueError, 'Cannot identify position of 5 in Set'):
        r.ord(5)
    r = RangeSet(ranges=(NR(2, 10, 2), NR(6, 12, 3)))
    for i, v in enumerate([2, 4, 6, 8, 9, 10, 12]):
        self.assertEqual(r.ord(v), i + 1)
        self.assertEqual(r[i + 1], v)
    with self.assertRaisesRegex(IndexError, 'Accessing Pyomo Sets by position is 1-based: valid Set positional index values are \\[1 .. len\\(Set\\)\\] or \\[-1 .. -len\\(Set\\)\\]'):
        r[0]
    with self.assertRaisesRegex(IndexError, 'FiniteScalarRangeSet index out of range'):
        r[10]
    with self.assertRaisesRegex(ValueError, 'Cannot identify position of 5 in Set'):
        r.ord(5)
    so = SetOf([0, (1,), 1])
    self.assertEqual(so.ord((1,)), 2)
    self.assertEqual(so.ord(1), 3)