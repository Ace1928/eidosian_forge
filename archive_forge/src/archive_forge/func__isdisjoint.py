import pickle
import pyomo.common.unittest as unittest
from pyomo.core.base.range import (
from pyomo.core.base.set import Any
def _isdisjoint(expected_result, a, b):
    self.assertIs(expected_result, a.isdisjoint(b))
    self.assertIs(expected_result, b.isdisjoint(a))