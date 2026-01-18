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
def test_set_skip(self):
    m = ConcreteModel()

    def _i_init(m, i):
        if i % 2:
            return Set.Skip
        return range(i)
    m.I = Set([1, 2, 3, 4, 5], initialize=_i_init)
    self.assertEqual(len(m.I), 2)
    self.assertIn(2, m.I)
    self.assertEqual(list(m.I[2]), [0, 1])
    self.assertIn(4, m.I)
    self.assertEqual(list(m.I[4]), [0, 1, 2, 3])
    self.assertNotIn(1, m.I)
    self.assertNotIn(3, m.I)
    self.assertNotIn(5, m.I)
    output = StringIO()
    m.I.pprint(ostream=output)
    ref = '\nI : Size=2, Index={1, 2, 3, 4, 5}, Ordered=Insertion\n    Key : Dimen : Domain : Size : Members\n      2 :     1 :    Any :    2 : {0, 1}\n      4 :     1 :    Any :    4 : {0, 1, 2, 3}\n'.strip()
    self.assertEqual(output.getvalue().strip(), ref.strip())
    m = ConcreteModel()

    def _i_init(m, i):
        if i % 2:
            return None
        return range(i)
    with self.assertRaisesRegex(ValueError, 'Set rule or initializer returned None instead of Set.Skip'):
        m.I = Set([1, 2, 3, 4, 5], initialize=_i_init)

    def _j_init(m):
        return None
    with self.assertRaisesRegex(ValueError, 'Set rule or initializer returned None instead of Set.Skip'):
        m.J = Set(initialize=_j_init)