from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableT(self):
    self.check_skiplist('tableT')
    with capture_output(currdir + 'loadT.dat'):
        print('load ' + self.filename('T') + ' format=transposed_array : T;')
    model = AbstractModel()
    model.B = Set(initialize=['I1', 'I2', 'I3', 'I4'])
    model.A = Set(initialize=['A1', 'A2', 'A3'])
    model.T = Param(model.A, model.B)
    instance = model.create_instance(currdir + 'loadT.dat')
    self.assertEqual(instance.T.extract_values(), {('A2', 'I1'): 2.3, ('A1', 'I2'): 1.4, ('A1', 'I3'): 1.5, ('A1', 'I4'): 1.6, ('A1', 'I1'): 1.3, ('A3', 'I4'): 3.6, ('A2', 'I4'): 2.6, ('A3', 'I1'): 3.3, ('A2', 'I3'): 2.5, ('A3', 'I2'): 3.4, ('A2', 'I2'): 2.4, ('A3', 'I3'): 3.5})
    os.remove(currdir + 'loadT.dat')