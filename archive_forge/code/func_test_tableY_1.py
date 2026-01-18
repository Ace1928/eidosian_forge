from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableY_1(self):
    with capture_output(currdir + 'loadY.dat'):
        print('table columns=2 Y(1)={2} := A1 3.3 A2 3.4 A3 3.5 ;')
    model = AbstractModel()
    model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
    model.Y = Param(model.A)
    instance = model.create_instance(currdir + 'loadY.dat')
    self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
    self.assertEqual(instance.Y.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
    os.remove(currdir + 'loadY.dat')