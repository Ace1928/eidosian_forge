import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Set, Param, Var, Objective
def Xtest_label2(self):
    model = AbstractModel()
    model.A = Set(initialize=[1, 2, 3])
    model.B = Param(model.A, initialize={1: 100, 2: 200, 3: 300})
    model.x = Var(model.A)
    model.y = Var(model.A)
    model.obj = Objective(rule=lambda inst: inst.x[1])
    instance = model.create_instance()
    instance.preprocess()
    self.assertEqual(instance.num_used_variables(), 1)
    self.assertEqual(instance.x[1].label, 'x(1)')
    self.assertEqual(instance.x[2].label, 'x(2)')
    self.assertEqual(instance.y[1].label, 'y(1)')