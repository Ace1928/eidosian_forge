from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableA1_2(self):
    with capture_output(currdir + 'loadA1.dat'):
        print('table A={A} : A := A1 A2 A3 ;')
    model = AbstractModel()
    model.A = Set()
    instance = model.create_instance(currdir + 'loadA1.dat')
    self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
    os.remove(currdir + 'loadA1.dat')