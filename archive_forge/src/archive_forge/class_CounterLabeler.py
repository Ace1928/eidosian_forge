import re
from pyomo.common.deprecation import deprecated
from pyomo.core.base.componentuid import ComponentUID
class CounterLabeler(object):

    def __init__(self, start=0):
        self._id = start

    def __call__(self, obj=None):
        self._id += 1
        return self._id