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
def _compare(self, other):
    """
        Compare *self* to *other* based on ``__name__`` and ``__module__``.

        Return 0 if they are equal, return 1 if *self* is
        greater than *other*, and return -1 if *self* is less than
        *other*.

        If *other* does not have ``__name__`` or ``__module__``, then
        return ``NotImplemented``.

        .. caution::
           This allows comparison to things well outside the type hierarchy,
           perhaps not symmetrically.

           For example, ``class Foo(object)`` and ``class Foo(Interface)``
           in the same file would compare equal, depending on the order of
           operands. Writing code like this by hand would be unusual, but it could
           happen with dynamic creation of types and interfaces.

        None is treated as a pseudo interface that implies the loosest
        contact possible, no contract. For that reason, all interfaces
        sort before None.
        """
    if other is self:
        return 0
    if other is None:
        return -1
    n1 = (self.__name__, self.__module__)
    try:
        n2 = (other.__name__, other.__module__)
    except AttributeError:
        return NotImplemented
    return (n1 > n2) - (n1 < n2)