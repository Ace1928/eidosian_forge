import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
def test_sparse_values(self):
    test = self.instance.A.sparse_values()
    self.assertEqual(type(test), list)
    test = zip(self.instance.A.keys(), test)
    self.validateDict(self.sparse_data.items(), test)