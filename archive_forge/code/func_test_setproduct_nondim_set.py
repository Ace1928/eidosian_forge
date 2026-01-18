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
def test_setproduct_nondim_set(self):
    m = ConcreteModel()
    m.I = Set(initialize=[1, 2, 3])
    m.J = Set()
    m.K = Set(initialize=[4, 5, 6])
    m.Z = m.I * m.J * m.K
    self.assertEqual(len(m.Z), 0)
    self.assertNotIn((2, 5), m.Z)
    m.J.add(0)
    self.assertEqual(len(m.Z), 9)
    self.assertIn((2, 0, 5), m.Z)