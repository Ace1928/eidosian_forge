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
def test_unconstructed_api(self):
    m = AbstractModel()
    m.I = Set(ordered=False)
    m.J = Set(ordered=Set.InsertionOrder)
    m.K = Set(ordered=Set.SortedOrder)
    with self.assertRaisesRegex(RuntimeError, "Cannot iterate over AbstractFiniteScalarSet 'I' before it has been constructed \\(initialized\\)"):
        for i in m.I:
            pass
    with self.assertRaisesRegex(RuntimeError, "Cannot iterate over AbstractOrderedScalarSet 'J' before it has been constructed \\(initialized\\)"):
        for i in m.J:
            pass
    with self.assertRaisesRegex(RuntimeError, "Cannot iterate over AbstractSortedScalarSet 'K' before it has been constructed \\(initialized\\)"):
        for i in m.K:
            pass
    with self.assertRaisesRegex(RuntimeError, "Cannot test membership in AbstractFiniteScalarSet 'I' before it has been constructed \\(initialized\\)"):
        1 in m.I
    with self.assertRaisesRegex(RuntimeError, "Cannot test membership in AbstractOrderedScalarSet 'J' before it has been constructed \\(initialized\\)"):
        1 in m.J
    with self.assertRaisesRegex(RuntimeError, "Cannot test membership in AbstractSortedScalarSet 'K' before it has been constructed \\(initialized\\)"):
        1 in m.K
    with self.assertRaisesRegex(RuntimeError, "Cannot access '__len__' on AbstractFiniteScalarSet 'I' before it has been constructed \\(initialized\\)"):
        len(m.I)
    with self.assertRaisesRegex(RuntimeError, "Cannot access '__len__' on AbstractOrderedScalarSet 'J' before it has been constructed \\(initialized\\)"):
        len(m.J)
    with self.assertRaisesRegex(RuntimeError, "Cannot access '__len__' on AbstractSortedScalarSet 'K' before it has been constructed \\(initialized\\)"):
        len(m.K)