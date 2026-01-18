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
def test_process_setarg(self):
    m = AbstractModel()
    m.I = Set([1, 2, 3])
    self.assertTrue(m.I.index_set().is_constructed())
    self.assertTrue(m.I.index_set().isordered())
    i = m.create_instance()
    self.assertEqual(i.I.index_set(), [1, 2, 3])
    m = AbstractModel()
    m.I = Set({1, 2, 3})
    self.assertTrue(m.I.index_set().is_constructed())
    self.assertFalse(m.I.index_set().isordered())
    i = m.create_instance()
    self.assertEqual(i.I.index_set(), [1, 2, 3])
    m = AbstractModel()
    m.I = Set(RangeSet(3))
    self.assertTrue(m.I.index_set().is_constructed())
    self.assertTrue(m.I.index_set().isordered())
    i = m.create_instance()
    self.assertEqual(i.I.index_set(), [1, 2, 3])
    m = AbstractModel()
    m.p = Param(initialize=3)
    m.I = Set(RangeSet(m.p))
    self.assertFalse(m.I.index_set().is_constructed())
    self.assertTrue(m.I.index_set().isordered())
    i = m.create_instance()
    self.assertEqual(i.I.index_set(), [1, 2, 3])
    m = AbstractModel()
    m.I = Set(lambda m: [1, 2, 3])
    self.assertFalse(m.I.index_set().is_constructed())
    self.assertTrue(m.I.index_set().isordered())
    i = m.create_instance()
    self.assertEqual(i.I.index_set(), [1, 2, 3])

    def _i_idx(m):
        return [1, 2, 3]
    m = AbstractModel()
    m.I = Set(_i_idx)
    self.assertFalse(m.I.index_set().is_constructed())
    self.assertTrue(m.I.index_set().isordered())
    i = m.create_instance()
    self.assertEqual(i.I.index_set(), [1, 2, 3])

    def _i_idx():
        yield 1
        yield 2
        yield 3
    m = Block()
    m.I = Set(_i_idx())
    self.assertFalse(m.I.index_set().is_constructed())
    self.assertTrue(m.I.index_set().isordered())
    i = ConcreteModel()
    i.m = m
    self.assertEqual(i.m.I.index_set(), [1, 2, 3])