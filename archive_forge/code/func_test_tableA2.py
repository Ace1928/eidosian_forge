from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableA2(self):
    self.check_skiplist('tableA2')
    with capture_output(currdir + 'loadA2.dat'):
        print('load ' + self.filename('A') + ' ;')
    model = AbstractModel()
    model.A = Set()
    try:
        instance = model.create_instance(currdir + 'loadA2.dat')
        self.fail('Should fail because no set name is specified')
    except IOError:
        pass
    except IndexError:
        pass
    os.remove(currdir + 'loadA2.dat')