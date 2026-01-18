import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
def tmp_init(model, i):
    tmp = []
    for i in range(0, value(model.n)):
        tmp.append(i / 2.0)
    return tmp