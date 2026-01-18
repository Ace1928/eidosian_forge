from io import StringIO
import os
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Set, BuildAction, value
def test_sparse_param_default(self):
    model = AbstractModel()
    model.Z = Set(initialize=[1, 3])
    model.A = Param(model.Z, initialize={1: 1.3}, default=0, mutable=True)
    model.action2 = BuildAction(model.Z, rule=action2_fn)
    instance = model.create_instance()
    tmp = value(instance.A[1])
    self.assertEqual(type(tmp), float)
    self.assertEqual(tmp, 2.3)