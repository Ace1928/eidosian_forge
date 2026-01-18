import unittest
import warnings
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
from traits.interface_checker import InterfaceError, check_implements
from traits import has_traits
def test_subclasses_with_wrong_signature_methods(self):
    """ Subclasses with incorrect method signatures """

    class IFoo(Interface):

        def foo(self, argument):
            pass

    @provides(IFoo)
    class Foo(HasTraits):

        def foo(self, argument):
            pass

    class Bar(Foo):

        def foo(self):
            pass
    self.assertRaises(InterfaceError, check_implements, Bar, IFoo, 2)