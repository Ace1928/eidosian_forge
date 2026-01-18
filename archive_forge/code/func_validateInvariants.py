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
def validateInvariants(self, obj, errors=None):
    """validate object to defined invariants."""
    for iface in self.__iro__:
        for invariant in iface.queryDirectTaggedValue('invariants', ()):
            try:
                invariant(obj)
            except Invalid as error:
                if errors is not None:
                    errors.append(error)
                else:
                    raise
    if errors:
        raise Invalid(errors)