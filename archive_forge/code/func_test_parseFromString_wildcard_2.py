import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_parseFromString_wildcard_2(self):
    cuid = ComponentUID('b[*,*].c.a[*]')
    self.assertEqual(cuid._cids, (('b', (_star, _star)), ('c', tuple()), ('a', (_star,))))