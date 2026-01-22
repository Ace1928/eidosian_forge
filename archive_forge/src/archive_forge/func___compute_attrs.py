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
def __compute_attrs(self, attrs):

    def update_value(aname, aval):
        if isinstance(aval, Attribute):
            aval.interface = self
            if not aval.__name__:
                aval.__name__ = aname
        elif isinstance(aval, FunctionType):
            aval = fromFunction(aval, self, name=aname)
        else:
            raise InvalidInterface('Concrete attribute, ' + aname)
        return aval
    return {aname: update_value(aname, aval) for aname, aval in attrs.items() if aname not in ('__locals__', '__qualname__', '__annotations__') and aval is not _decorator_non_return}