import itertools
from types import FunctionType
from zope.interface import Interface
from zope.interface import classImplements
from zope.interface.interface import InterfaceClass
from zope.interface.interface import _decorator_non_return
from zope.interface.interface import fromFunction
def _create_ABCInterface():
    abc_name_bases_attrs = ('ABCInterface', (Interface,), {})
    instance = ABCInterfaceClass.__new__(ABCInterfaceClass, *abc_name_bases_attrs)
    InterfaceClass.__init__(instance, *abc_name_bases_attrs)
    return instance