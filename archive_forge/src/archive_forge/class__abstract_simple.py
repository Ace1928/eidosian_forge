import inspect
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.core.base.disable_methods import disable_methods
from pyomo.common.modeling import NOTSET
@disable_methods(('a', ('b', 'custom_msg'), 'd', ('e', 'custom_pmsg'), 'f', 'g', ('h', 'custom_pmsg')))
class _abstract_simple(_simple):
    pass