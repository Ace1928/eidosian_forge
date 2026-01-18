import pickle
import os
from os.path import abspath, dirname, join
import platform
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_pickle_concrete_model_constraint(self):
    model = ConcreteModel()
    model.x = Var()
    model.A = Constraint(expr=model.x <= 0)
    str = pickle.dumps(model)
    tmodel = pickle.loads(str)
    self.verifyModel(model, tmodel)