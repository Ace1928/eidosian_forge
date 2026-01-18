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
def test_tricross_set(self):
    self.model.D = self.model.A * self.model.B * self.model.C
    self.instance = self.model.create_instance()
    self.assertEqual(len(self.instance.D), 27)