import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.common.collections import ComponentSet
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set import (
from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import (
def test_simple_lookup(self):
    m = self.m
    self._lookupTester(m.b[:, :].x[:, :], (1, 5, 7, 10), m.b[1, 5].x[7, 10])
    self._lookupTester(m.b[:, 4].x[8, :], (1, 10), m.b[1, 4].x[8, 10])
    self._lookupTester(m.b[:, 4].x[8, 10], (1,), m.b[1, 4].x[8, 10])
    self._lookupTester(m.b[1, 4].x[8, :], (10,), m.b[1, 4].x[8, 10])
    self._lookupTester(m.b[:, :].y[:], (1, 5, 7), m.b[1, 5].y[7])
    self._lookupTester(m.b[:, 4].y[:], (1, 7), m.b[1, 4].y[7])
    self._lookupTester(m.b[:, 4].y[8], (1,), m.b[1, 4].y[8])
    self._lookupTester(m.b[:, :].z, (1, 5), m.b[1, 5].z)
    self._lookupTester(m.b[:, 4].z, (1,), m.b[1, 4].z)
    self._lookupTester(m.c[:].x[:, :], (1, 7, 10), m.c[1].x[7, 10])
    self._lookupTester(m.c[:].x[8, :], (1, 10), m.c[1].x[8, 10])
    self._lookupTester(m.c[:].x[8, 10], (1,), m.c[1].x[8, 10])
    self._lookupTester(m.c[1].x[:, :], (8, 10), m.c[1].x[8, 10])
    self._lookupTester(m.c[1].x[8, :], (10,), m.c[1].x[8, 10])
    self._lookupTester(m.c[:].y[:], (1, 7), m.c[1].y[7])
    self._lookupTester(m.c[:].y[8], (1,), m.c[1].y[8])
    self._lookupTester(m.c[1].y[:], (8,), m.c[1].y[8])
    self._lookupTester(m.c[:].z, (1,), m.c[1].z)
    m.jagged_set = Set(initialize=[1, (2, 3)], dimen=None)
    m.jb = Block(m.jagged_set)
    m.jb[1].x = Var([1, 2, 3])
    m.jb[2, 3].x = Var([1, 2, 3])
    self._lookupTester(m.jb[...], (1,), m.jb[1])
    self._lookupTester(m.jb[...].x[:], (1, 2), m.jb[1].x[2])
    self._lookupTester(m.jb[...].x[:], (2, 3, 2), m.jb[2, 3].x[2])
    rd = _ReferenceDict(m.jb[:, :, :].x[:])
    with self.assertRaises(KeyError):
        rd[2, 3, 4, 2]
    rd = _ReferenceDict(m.b[:, 4].x[:])
    with self.assertRaises(KeyError):
        rd[1, 0]