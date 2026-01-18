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
def test_single_set(self):
    tmp = Set()
    a = SetInitializer(None)
    self.assertIs(type(a), SetInitializer)
    self.assertIsNone(a._set)
    self.assertIs(a(None, None, tmp), Any)
    self.assertTrue(a.constant())
    self.assertFalse(a.verified)
    a = SetInitializer(Reals)
    self.assertIs(type(a), SetInitializer)
    self.assertIs(type(a._set), ConstantInitializer)
    self.assertIs(a(None, None, tmp), Reals)
    self.assertIs(a._set.val, Reals)
    self.assertTrue(a.constant())
    self.assertFalse(a.verified)
    a = SetInitializer({1: Reals})
    self.assertIs(type(a), SetInitializer)
    self.assertIs(type(a._set), ItemInitializer)
    self.assertIs(a(None, 1, tmp), Reals)
    self.assertFalse(a.constant())
    self.assertFalse(a.verified)