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
@_use_c_impl
class ObjectSpecificationDescriptor:
    """Implement the ``__providedBy__`` attribute

    The ``__providedBy__`` attribute computes the interfaces provided by
    an object. If an object has an ``__provides__`` attribute, that is returned.
    Otherwise, `implementedBy` the *cls* is returned.

    .. versionchanged:: 5.4.0
       Both the default (C) implementation and the Python implementation
       now let exceptions raised by accessing ``__provides__`` propagate.
       Previously, the C version ignored all exceptions.
    .. versionchanged:: 5.4.0
       The Python implementation now matches the C implementation and lets
       a ``__provides__`` of ``None`` override what the class is declared to
       implement.
    """

    def __get__(self, inst, cls):
        """Get an object specification for an object
        """
        if inst is None:
            return getObjectSpecification(cls)
        try:
            return inst.__provides__
        except AttributeError:
            return implementedBy(cls)