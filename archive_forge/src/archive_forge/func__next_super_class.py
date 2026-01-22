import sys
import weakref
from types import FunctionType
from types import MethodType
from types import ModuleType
from zope.interface._compat import _use_c_impl
from zope.interface.interface import Interface
from zope.interface.interface import InterfaceClass
from zope.interface.interface import NameAndModuleComparisonMixin
from zope.interface.interface import Specification
from zope.interface.interface import SpecificationBase
def _next_super_class(ob):
    self_class = ob.__self_class__
    class_that_invoked_super = ob.__thisclass__
    complete_mro = self_class.__mro__
    next_class = complete_mro[complete_mro.index(class_that_invoked_super) + 1]
    return next_class