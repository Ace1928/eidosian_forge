import datetime
import multiprocessing
import os
import time
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Param
@unittest.timeout(10)
def test_timeout_skip(self):
    if TestPyomoUnittest.test_timeout_skip.skip:
        self.skipTest('Skipping this test')
    self.assertEqual(0, 1)