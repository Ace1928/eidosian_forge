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
def test_other2(self):
    self.model.Z = Set(initialize=['A'])
    self.model.A = Set(self.model.Z, initialize={'A': [1, 2, 3, 'A']}, within=Integers)
    try:
        self.instance = self.model.create_instance()
    except ValueError:
        pass
    else:
        self.fail('fail test_other1')