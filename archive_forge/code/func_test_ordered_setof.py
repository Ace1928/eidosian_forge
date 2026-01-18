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
def test_ordered_setof(self):
    i = SetOf([1, 3, 2, 0])
    self.assertTrue(i.isfinite())
    self.assertTrue(i.isordered())
    self.assertEqual(i.ordered_data(), (1, 3, 2, 0))
    self.assertEqual(i.sorted_data(), (0, 1, 2, 3))
    self.assertEqual(tuple(reversed(i)), (0, 2, 3, 1))
    self.assertEqual(i[2], 3)
    self.assertEqual(i[-1], 0)
    with self.assertRaisesRegex(IndexError, 'Accessing Pyomo Sets by position is 1-based: valid Set positional index values are \\[1 .. len\\(Set\\)\\] or \\[-1 .. -len\\(Set\\)\\]'):
        i[0]
    with self.assertRaisesRegex(IndexError, 'OrderedSetOf index out of range'):
        i[5]
    with self.assertRaisesRegex(IndexError, 'OrderedSetOf index out of range'):
        i[-5]
    self.assertEqual(i.ord(3), 2)
    with self.assertRaisesRegex(ValueError, '5 is not in list'):
        i.ord(5)
    self.assertEqual(i.first(), 1)
    self.assertEqual(i.last(), 0)
    self.assertEqual(i.next(3), 2)
    self.assertEqual(i.prev(2), 3)
    self.assertEqual(i.nextw(3), 2)
    self.assertEqual(i.prevw(2), 3)
    self.assertEqual(i.next(3, 2), 0)
    self.assertEqual(i.prev(2, 2), 1)
    self.assertEqual(i.nextw(3, 2), 0)
    self.assertEqual(i.prevw(2, 2), 1)
    with self.assertRaisesRegex(IndexError, 'Cannot advance past the end of the Set'):
        i.next(0)
    with self.assertRaisesRegex(IndexError, 'Cannot advance before the beginning of the Set'):
        i.prev(1)
    self.assertEqual(i.nextw(0), 1)
    self.assertEqual(i.prevw(1), 0)
    with self.assertRaisesRegex(IndexError, 'Cannot advance past the end of the Set'):
        i.next(0, 2)
    with self.assertRaisesRegex(IndexError, 'Cannot advance before the beginning of the Set'):
        i.prev(1, 2)
    self.assertEqual(i.nextw(0, 2), 3)
    self.assertEqual(i.prevw(1, 2), 2)
    i = SetOf((1, 3, 2, 0))
    self.assertTrue(i.isfinite())
    self.assertTrue(i.isordered())
    self.assertEqual(i.ordered_data(), (1, 3, 2, 0))
    self.assertEqual(i.sorted_data(), (0, 1, 2, 3))
    self.assertEqual(tuple(reversed(i)), (0, 2, 3, 1))
    self.assertEqual(i[2], 3)
    self.assertEqual(i[-1], 0)
    with self.assertRaisesRegex(IndexError, 'Accessing Pyomo Sets by position is 1-based: valid Set positional index values are \\[1 .. len\\(Set\\)\\] or \\[-1 .. -len\\(Set\\)\\]'):
        i[0]
    with self.assertRaisesRegex(IndexError, 'OrderedSetOf index out of range'):
        i[5]
    with self.assertRaisesRegex(IndexError, 'OrderedSetOf index out of range'):
        i[-5]
    self.assertEqual(i.ord(3), 2)
    with self.assertRaisesRegex(ValueError, 'x not in tuple'):
        i.ord(5)
    i = SetOf([1, None, 'a'])
    self.assertTrue(i.isfinite())
    self.assertTrue(i.isordered())
    self.assertEqual(i.ordered_data(), (1, None, 'a'))
    self.assertEqual(i.sorted_data(), (None, 1, 'a'))
    self.assertEqual(tuple(reversed(i)), ('a', None, 1))