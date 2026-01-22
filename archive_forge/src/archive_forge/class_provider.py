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
class provider:
    """Declare interfaces provided directly by a class

      This function is called in a class definition.

      The arguments are one or more interfaces or interface specifications
      (`~zope.interface.interfaces.IDeclaration` objects).

      The given interfaces (including the interfaces in the specifications)
      are used to create the class's direct-object interface specification.
      An error will be raised if the module class has an direct interface
      specification. In other words, it is an error to call this function more
      than once in a class definition.

      Note that the given interfaces have nothing to do with the interfaces
      implemented by instances of the class.

      This function is provided for convenience. It provides a more convenient
      way to call `directlyProvides` for a class. For example::

        @provider(I1)
        class C:
            pass

      is equivalent to calling::

        directlyProvides(C, I1)

      after the class has been created.
    """

    def __init__(self, *interfaces):
        self.interfaces = interfaces

    def __call__(self, ob):
        directlyProvides(ob, *self.interfaces)
        return ob