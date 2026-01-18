import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
def test_provides_with_interface_check_warn(self):

    class Test(HasTraits):
        pass
    provides_ifoo = provides(IFoo)
    with self.set_check_interfaces(1):
        with self.assertWarns(DeprecationWarning) as warnings_cm:
            with self.assertLogs('traits', logging.WARNING):
                Test = provides_ifoo(Test)
    test = Test()
    self.assertIsInstance(test, IFoo)
    self.assertIn('the @provides decorator will not perform interface checks', str(warnings_cm.warning))
    _, _, this_module = __name__.rpartition('.')
    self.assertIn(this_module, warnings_cm.filename)