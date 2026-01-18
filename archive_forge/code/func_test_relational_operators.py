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
def test_relational_operators(self):
    Any2 = _AnySet()
    self.assertTrue(Any.issubset(Any2))
    self.assertTrue(Any.issuperset(Any2))
    self.assertFalse(Any.isdisjoint(Any2))
    Reals2 = RangeSet(ranges=(NR(None, None, 0),))
    self.assertTrue(Reals.issubset(Reals2))
    self.assertTrue(Reals.issuperset(Reals2))
    self.assertFalse(Reals.isdisjoint(Reals2))
    Integers2 = RangeSet(ranges=(NR(0, None, -1), NR(0, None, 1)))
    self.assertTrue(Integers.issubset(Integers2))
    self.assertTrue(Integers.issuperset(Integers2))
    self.assertFalse(Integers.isdisjoint(Integers2))
    self.assertTrue(Integers.issubset(Reals))
    self.assertFalse(Integers.issuperset(Reals))
    self.assertFalse(Integers.isdisjoint(Reals))
    self.assertFalse(Reals.issubset(Integers))
    self.assertTrue(Reals.issuperset(Integers))
    self.assertFalse(Reals.isdisjoint(Integers))
    self.assertTrue(Reals.issubset(Any))
    self.assertFalse(Reals.issuperset(Any))
    self.assertFalse(Reals.isdisjoint(Any))
    self.assertFalse(Any.issubset(Reals))
    self.assertTrue(Any.issuperset(Reals))
    self.assertFalse(Any.isdisjoint(Reals))
    self.assertFalse(Integers.issubset(PositiveIntegers))
    self.assertTrue(Integers.issuperset(PositiveIntegers))
    self.assertFalse(Integers.isdisjoint(PositiveIntegers))
    self.assertTrue(PositiveIntegers.issubset(Integers))
    self.assertFalse(PositiveIntegers.issuperset(Integers))
    self.assertFalse(PositiveIntegers.isdisjoint(Integers))
    tmp = IntegerSet()
    tmp.clear()
    self.assertTrue(tmp.issubset(EmptySet))
    self.assertTrue(tmp.issuperset(EmptySet))
    self.assertTrue(tmp.isdisjoint(EmptySet))
    self.assertTrue(EmptySet.issubset(tmp))
    self.assertTrue(EmptySet.issuperset(tmp))
    self.assertTrue(EmptySet.isdisjoint(tmp))