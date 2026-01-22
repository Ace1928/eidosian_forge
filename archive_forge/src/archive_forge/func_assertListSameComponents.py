import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def assertListSameComponents(self, m, cuid1, cuid2):
    self.assertTrue(cuid1.list_components(m))
    self.assertEqual(len(list(cuid1.list_components(m))), len(list(cuid2.list_components(m))))
    for c1, c2 in zip(cuid1.list_components(m), cuid2.list_components(m)):
        self.assertIs(c1, c2)