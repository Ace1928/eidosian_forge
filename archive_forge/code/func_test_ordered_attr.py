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
def test_ordered_attr(self):
    m = ConcreteModel()
    m.J = Set(ordered=True)
    m.K = Set(ordered=False)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        self.assertTrue(m.J.ordered)
    self.assertRegex(output.getvalue(), "^DEPRECATED: The 'ordered' attribute is no longer supported.")
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        self.assertFalse(m.K.ordered)
    self.assertRegex(output.getvalue(), "^DEPRECATED: The 'ordered' attribute is no longer supported.")