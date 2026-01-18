from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tablePP(self):
    self.check_skiplist('tablePP')
    dp = DataPortal()
    dp.load(param='PP', **self.create_options('PP'))
    self.assertEqual(dp.data('PP'), {('A3', 'B3'): 4.5, ('A1', 'B1'): 4.3, ('A2', 'B2'): 4.4})