from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableXW_3_1(self):
    with capture_output(currdir + 'loadXW.dat'):
        print('table columns=3 A={1} X(A)={2} W(A)={3} := A1 3.3 4.3 A2 3.4 4.4 A3 3.5 4.5 ;')
    model = AbstractModel()
    model.A = Set()
    model.X = Param(model.A)
    model.W = Param(model.A)
    instance = model.create_instance(currdir + 'loadXW.dat')
    self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
    self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
    self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
    os.remove(currdir + 'loadXW.dat')