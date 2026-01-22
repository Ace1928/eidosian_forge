import pickle
import pyomo.common.unittest as unittest
from pyomo.core.base.range import (
from pyomo.core.base.set import Any
class _NonComparable(_Unrelated):

    def __init__(self, val):
        self.val = val

    def __sub__(self, other):
        return self

    def __gt__(self, other):
        return True

    def __le__(self, other):
        return True