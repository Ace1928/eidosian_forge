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
@staticmethod
def _add_interfaces_to_cls(interfaces, cls):
    implemented_by_cls = implementedBy(cls)
    interfaces = tuple([iface for iface in interfaces if not implemented_by_cls.isOrExtends(iface)])
    return interfaces + (implemented_by_cls,)