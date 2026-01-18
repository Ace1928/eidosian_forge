from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableA1_3(self):
    model = AbstractModel()
    model.A = Set()
    data = DataPortal()
    data.connect(filename=os.path.abspath(example_dir + 'B.tab'))
    data.connect(filename=os.path.abspath(example_dir + 'A.tab'))
    data.load(set=model.A)
    data.disconnect()
    instance = model.create_instance(data)
    self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))