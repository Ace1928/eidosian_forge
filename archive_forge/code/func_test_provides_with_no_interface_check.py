import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
def test_provides_with_no_interface_check(self):

    class Test(HasTraits):
        pass
    provides_ifoo = provides(IFoo)
    with self.set_check_interfaces(0):
        Test = provides_ifoo(Test)
    test = Test()
    self.assertIsInstance(test, IFoo)