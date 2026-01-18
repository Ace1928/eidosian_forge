from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tablePO_1(self):
    with capture_output(currdir + 'loadPO.dat'):
        print('table columns=4 J={1,2} P(J)={3} O(J)={4} := A1 B1 4.3 5.3 A2 B2 4.4 5.4 A3 B3 4.5 5.5 ;')
    model = AbstractModel()
    model.J = Set(dimen=2)
    model.P = Param(model.J)
    model.O = Param(model.J)
    instance = model.create_instance(currdir + 'loadPO.dat')
    self.assertEqual(set(instance.J.data()), set([('A3', 'B3'), ('A1', 'B1'), ('A2', 'B2')]))
    self.assertEqual(instance.P.extract_values(), {('A3', 'B3'): 4.5, ('A1', 'B1'): 4.3, ('A2', 'B2'): 4.4})
    self.assertEqual(instance.O.extract_values(), {('A3', 'B3'): 5.5, ('A1', 'B1'): 5.3, ('A2', 'B2'): 5.4})
    os.remove(currdir + 'loadPO.dat')