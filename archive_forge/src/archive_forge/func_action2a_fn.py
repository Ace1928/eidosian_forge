import os
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, BuildCheck, Param, Set, value
def action2a_fn(model, i):
    ans = True
    if i in model.A:
        return value(model.A[i]) == 1.3
    return True