from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableB(self):
    self.check_skiplist('tableB')
    model = AbstractModel()
    model.B = Set()
    data = DataPortal()
    data.load(set=model.B, **self.create_options('B'))
    instance = model.create_instance(data)
    self.assertEqual(set(instance.B.data()), set([1, 2, 3]))