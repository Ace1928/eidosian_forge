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
class implementer:
    """
    Declare the interfaces implemented by instances of a class.

    This function is called as a class decorator.

    The arguments are one or more interfaces or interface
    specifications (`~zope.interface.interfaces.IDeclaration`
    objects).

    The interfaces given (including the interfaces in the
    specifications) are added to any interfaces previously declared,
    unless the interface is already implemented.

    Previous declarations include declarations for base classes unless
    implementsOnly was used.

    This function is provided for convenience. It provides a more
    convenient way to call `classImplements`. For example::

        @implementer(I1)
        class C(object):
            pass

    is equivalent to calling::

        classImplements(C, I1)

    after the class has been created.

    .. seealso:: `classImplements`
       The change history provided there applies to this function too.
    """
    __slots__ = ('interfaces',)

    def __init__(self, *interfaces):
        self.interfaces = interfaces

    def __call__(self, ob):
        if isinstance(ob, type):
            classImplements(ob, *self.interfaces)
            return ob
        spec_name = _implements_name(ob)
        spec = Implements.named(spec_name, *self.interfaces)
        try:
            ob.__implemented__ = spec
        except AttributeError:
            raise TypeError("Can't declare implements", ob)
        return ob