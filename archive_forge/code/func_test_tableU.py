from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableU(self):
    self.check_skiplist('tableU')
    with capture_output(currdir + 'loadU.dat'):
        print('load ' + self.filename('U') + ' format=array : U;')
    model = AbstractModel()
    model.A = Set(initialize=['I1', 'I2', 'I3', 'I4'])
    model.B = Set(initialize=['A1', 'A2', 'A3'])
    model.U = Param(model.A, model.B)
    instance = model.create_instance(currdir + 'loadU.dat')
    self.assertEqual(instance.U.extract_values(), {('I2', 'A1'): 1.4, ('I3', 'A1'): 1.5, ('I3', 'A2'): 2.5, ('I4', 'A1'): 1.6, ('I3', 'A3'): 3.5, ('I1', 'A2'): 2.3, ('I4', 'A3'): 3.6, ('I1', 'A3'): 3.3, ('I4', 'A2'): 2.6, ('I2', 'A3'): 3.4, ('I1', 'A1'): 1.3, ('I2', 'A2'): 2.4})
    os.remove(currdir + 'loadU.dat')