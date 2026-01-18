import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_nonindexed_block_immutable_param(self):
    model = AbstractModel()

    def _b_rule(b):
        b.A = Param(initialize=2.0)
    model.B = Block(rule=_b_rule)
    instance = model.create_instance()
    self.assertEqual(value(instance.B.A), 2.0)