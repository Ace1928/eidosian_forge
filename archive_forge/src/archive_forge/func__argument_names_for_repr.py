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
def _argument_names_for_repr(interfaces):
    ordered_names = []
    names = set()
    for iface in interfaces:
        duplicate_transform = repr
        if isinstance(iface, InterfaceClass):
            this_name = iface.__name__
            duplicate_transform = str
        elif isinstance(iface, type):
            this_name = iface.__name__
            duplicate_transform = _implements_name
        elif isinstance(iface, Implements) and (not iface.declared) and (iface.inherit in interfaces):
            continue
        else:
            this_name = repr(iface)
        already_seen = this_name in names
        names.add(this_name)
        if already_seen:
            this_name = duplicate_transform(iface)
        ordered_names.append(this_name)
    return ', '.join(ordered_names)