import pickle
import os
from os.path import abspath, dirname, join
import platform
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_pickle_abstract_model_mutable_indexed_param(self):
    model = AbstractModel()
    model.A = Param([1, 2, 3], initialize={1: 100, 3: 300}, mutable=True)
    str = pickle.dumps(model)
    tmodel = pickle.loads(str)
    self.verifyModel(model, tmodel)