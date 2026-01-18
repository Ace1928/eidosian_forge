import pickle
import os
from os.path import abspath, dirname, join
import platform
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_pickle_abstract_model_var(self):
    model = AbstractModel()
    model.A = Var(initialize=1)
    str = pickle.dumps(model)
    tmodel = pickle.loads(str)
    self.verifyModel(model, tmodel)