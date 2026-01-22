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
def _lookupTester(self, _slice, key, ans):
    rd = _ReferenceDict(_slice)
    self.assertIn(key, rd)
    self.assertIs(rd[key], ans)
    if len(key) == 1:
        self.assertIn(key[0], rd)
        self.assertIs(rd[key[0]], ans)
    self.assertNotIn(None, rd)
    with self.assertRaises(KeyError):
        rd[None]
    for i in range(len(key)):
        _ = tuple([0] * i)
        self.assertNotIn(_, rd)
        with self.assertRaises(KeyError):
            rd[_]