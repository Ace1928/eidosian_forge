from io import StringIO
import os
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Set, BuildAction, value
def action3_fn(model, i):
    if i in model.A.sparse_keys():
        model.A[i] = value(model.A[i]) + i