from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableC(self):
    self.check_skiplist('tableC')
    with capture_output(currdir + 'loadC.dat'):
        print('load ' + self.filename('C') + ' format=set : C ;')
    model = AbstractModel()
    model.C = Set(dimen=2)
    instance = model.create_instance(currdir + 'loadC.dat')
    self.assertEqual(set(instance.C.data()), set([('A1', 1), ('A1', 2), ('A1', 3), ('A2', 1), ('A2', 2), ('A2', 3), ('A3', 1), ('A3', 2), ('A3', 3)]))
    os.remove(currdir + 'loadC.dat')