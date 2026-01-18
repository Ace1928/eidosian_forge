from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableB1_1(self):
    with capture_output(currdir + 'loadB.dat'):
        print('table columns=1 B={1} := 1 2 3 ;')
    model = AbstractModel()
    model.B = Set()
    instance = model.create_instance(currdir + 'loadB.dat')
    self.assertEqual(set(instance.B.data()), set([1, 2, 3]))
    os.remove(currdir + 'loadB.dat')