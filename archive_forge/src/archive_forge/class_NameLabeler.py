import re
from pyomo.common.deprecation import deprecated
from pyomo.core.base.componentuid import ComponentUID
class NameLabeler(object):

    def __call__(self, obj):
        return obj.getname(True)