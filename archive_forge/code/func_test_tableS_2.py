from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableS_2(self):
    with capture_output(currdir + 'loadS.dat'):
        print('table S(A) : A S := A1 3.3 A2 . A3 3.5 ;')
    model = AbstractModel()
    model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
    model.S = Param(model.A)
    instance = model.create_instance(currdir + 'loadS.dat')
    self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
    self.assertEqual(instance.S.extract_values(), {'A1': 3.3, 'A3': 3.5})
    os.remove(currdir + 'loadS.dat')