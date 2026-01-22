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
class ArrayParam_mutable_dense_intDefault_denseParamInit(ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        self.model = AbstractModel()
        self.model.p = Param([1, 3], initialize={1: 1.3, 3: 2.3})
        ParamTester.setUp(self, mutable=True, initialize=self.model.p, default=99.5, **kwds)
        self.sparse_data = {1: 1.3, 3: 2.3}
        self.data = self.sparse_data