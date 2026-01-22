import re
from pyomo.common.deprecation import deprecated
from pyomo.core.base.componentuid import ComponentUID
class CuidLabeler(object):

    def __call__(self, obj=None):
        return ComponentUID(obj)