from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_store_param4(self):
    self.check_skiplist('store_param4')
    model = ConcreteModel()
    model.A = Set(initialize=[(1, 2), (2, 3), (3, 4)], dimen=2)
    model.p = Param(model.A, initialize={(1, 2): 10, (2, 3): 20, (3, 4): 30})
    model.q = Param(model.A, initialize={(1, 2): 11, (2, 3): 21, (3, 4): 31})
    data = DataPortal()
    data.store(param=(model.p, model.q), **self.create_write_options('param4'))
    self.compare_data('param4', self.suffix)