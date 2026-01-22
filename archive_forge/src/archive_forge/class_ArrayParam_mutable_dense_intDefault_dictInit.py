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
class ArrayParam_mutable_dense_intDefault_dictInit(ParamTester, unittest.TestCase):

    def setUp(self, **kwds):

        def A_init(model, i):
            return 1.5 + i
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=True, initialize=A_init, **kwds)
        self.sparse_data = {1: 2.5, 3: 4.5}
        self.data = self.sparse_data