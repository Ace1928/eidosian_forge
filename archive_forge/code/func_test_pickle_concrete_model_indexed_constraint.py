import pickle
import os
from os.path import abspath, dirname, join
import platform
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_pickle_concrete_model_indexed_constraint(self):
    model = ConcreteModel()
    model.x = Var()
    model.A = Constraint([1, 2, 3], rule=simple_con_rule)
    str = pickle.dumps(model)
    tmodel = pickle.loads(str)
    self.verifyModel(model, tmodel)