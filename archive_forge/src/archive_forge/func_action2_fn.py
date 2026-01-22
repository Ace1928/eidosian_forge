from io import StringIO
import os
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Set, BuildAction, value
def action2_fn(model, i):
    if i in model.A:
        model.A[i] = value(model.A[i]) + i