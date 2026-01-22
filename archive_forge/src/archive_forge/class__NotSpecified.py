from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.environ import (
from pyomo.core.base.set import GlobalSets
class _NotSpecified(object):
    pass