import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_parseFromRepr1_typeID(self):
    cuid = ComponentUID('b:#1,$2.c.a:2')
    self.assertEqual(cuid._cids, (('b', (1, '2')), ('c', tuple()), ('a', (2,))))