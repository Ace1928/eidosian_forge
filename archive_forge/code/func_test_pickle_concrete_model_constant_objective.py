import pickle
import os
from os.path import abspath, dirname, join
import platform
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_pickle_concrete_model_constant_objective(self):
    model = ConcreteModel()
    model.A = Objective(expr=1)
    str = pickle.dumps(model)
    tmodel = pickle.loads(str)
    self.verifyModel(model, tmodel)