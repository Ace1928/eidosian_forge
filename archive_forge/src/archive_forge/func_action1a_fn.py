import os
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, BuildCheck, Param, Set, value
def action1a_fn(model):
    return value(model.A) == 3.3