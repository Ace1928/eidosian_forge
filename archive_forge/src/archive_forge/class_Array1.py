import os
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, BuildCheck, Param, Set, value
class Array1(PyomoModel):

    def setUp(self):
        PyomoModel.setUp(self)
        self.model.Z = Set(initialize=[1, 3])
        self.model.A = Param(self.model.Z, initialize=1.3)

    def tearDown(self):
        PyomoModel.tearDown(self)

    def test_true(self):
        """Check the value of the parameter"""
        self.model.action2 = BuildCheck(self.model.Z, rule=action2a_fn)
        self.instance = self.model.create_instance()

    def test_false(self):
        """Check the value of the parameter"""
        self.model.action2 = BuildCheck(self.model.Z, rule=action2b_fn)
        try:
            self.instance = self.model.create_instance()
            self.fail('expected failure')
        except ValueError:
            pass