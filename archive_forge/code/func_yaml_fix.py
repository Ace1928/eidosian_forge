import copy
import enum
from io import StringIO
from math import inf
from pyomo.common.collections import Bunch
def yaml_fix(self, val):
    if not isinstance(val, str):
        return val
    return val.replace(':', '\\x3a')