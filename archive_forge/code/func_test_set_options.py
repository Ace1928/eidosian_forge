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
def test_set_options(self):
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):

        @set_options(domain=Integers)
        def Bindex(m):
            return range(5)
    self.assertIn('The set_options decorator is deprecated', output.getvalue())
    m = ConcreteModel()
    m.I = Set(initialize=[8, 9])
    m.J = m.I.cross(Bindex)
    self.assertIs(m.J._sets[1]._domain, Integers)
    m.K = Set(Bindex)
    self.assertIs(m.K.index_set()._domain, Integers)
    self.assertEqual(m.K.index_set(), [0, 1, 2, 3, 4])