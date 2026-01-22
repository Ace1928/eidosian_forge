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
def _verify_finite_difference(self, a, b):
    if isinstance(a, (Set, SetOf, RangeSet)):
        a_finite = a.isfinite()
    else:
        a_finite = True
    if isinstance(b, (Set, SetOf, RangeSet)):
        b_finite = b.isfinite()
    else:
        b_finite = True
    self.assertTrue(a_finite or b_finite)
    x = a - b
    self.assertIs(type(x), SetDifference_FiniteSet)
    self.assertTrue(x.isfinite())
    self.assertFalse(x.isordered())
    self.assertEqual(len(x), 3)
    self.assertEqual(sorted(list(x)), [2, 3, 5])
    self.assertEqual(x.ordered_data(), (2, 3, 5))
    self.assertEqual(x.sorted_data(), (2, 3, 5))
    self.assertNotIn(0, x)
    self.assertNotIn(1, x)
    self.assertIn(2, x)
    self.assertIn(3, x)
    self.assertNotIn(4, x)
    self.assertIn(5, x)
    self.assertNotIn(6, x)
    self.assertEqual(len(list(x._sets[0].ranges()) + list(x._sets[1].ranges())), 9)
    self.assertEqual(len(list(x.ranges())), 3)