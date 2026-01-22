import sys
import weakref
from types import FunctionType
from types import MethodType
from typing import Union
from zope.interface import ro
from zope.interface._compat import _use_c_impl
from zope.interface.exceptions import Invalid
from zope.interface.ro import ro as calculate_ro
from zope.interface.declarations import implementedBy
from zope.interface.declarations import providedBy
from zope.interface.exceptions import BrokenImplementation
from zope.interface.exceptions import InvalidInterface
from zope.interface.declarations import _empty
class _InterfaceMetaClass(type):
    __slots__ = ()

    def __new__(cls, name, bases, attrs):
        __module__ = sys._getframe(1).f_globals['__name__']
        moduledescr = InterfaceBase.__dict__['__module__']
        if isinstance(moduledescr, str):
            moduledescr = InterfaceBase.__dict__['__module_property__']
        attrs['__module__'] = moduledescr
        kind = type.__new__(cls, name, bases, attrs)
        kind.__module = __module__
        return kind

    @property
    def __module__(cls):
        return cls.__module

    def __repr__(cls):
        return "<class '{}.{}'>".format(cls.__module, cls.__name__)