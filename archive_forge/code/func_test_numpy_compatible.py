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
@unittest.skipIf(not numpy_available, 'NumPy required for these tests')
def test_numpy_compatible(self):
    self.assertIn(np.intc(1), Reals)
    self.assertIn(np.float64(1), Reals)
    self.assertIn(np.float64(1.5), Reals)
    self.assertIn(np.intc(1), Integers)
    self.assertIn(np.float64(1), Integers)
    self.assertNotIn(np.float64(1.5), Integers)