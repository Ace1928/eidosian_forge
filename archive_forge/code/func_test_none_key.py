import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_none_key(self):
    model = AbstractModel()
    model.b = Block()
    inst = model.create_instance()
    self.assertEqual(id(inst.b), id(inst.b[None]))