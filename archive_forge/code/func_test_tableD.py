from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableD(self):
    self.check_skiplist('tableD')
    with capture_output(currdir + 'loadD.dat'):
        print('load ' + self.filename('D') + ' format=set_array : C ;')
    model = AbstractModel()
    model.C = Set(dimen=2)
    instance = model.create_instance(currdir + 'loadD.dat')
    self.assertEqual(set(instance.C.data()), set([('A1', 1), ('A2', 2), ('A3', 3)]))
    os.remove(currdir + 'loadD.dat')