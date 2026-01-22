from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.fbbt.tests.test_fbbt import FbbtTestBase
from pyomo.common.errors import InfeasibleConstraintException
import math
@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
class TestFbbt(FbbtTestBase, unittest.TestCase):

    def setUp(self) -> None:
        self.it = appsi.fbbt.IntervalTightener()
        self.tightener = self.it.perform_fbbt