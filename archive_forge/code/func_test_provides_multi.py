import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
def test_provides_multi(self):

    @provides(IFoo, IAverage, IList)
    class Test(HasTraits):

        def get_foo(self):
            return 'test_foo'

        def get_average(self):
            return 42

        def get_list(self):
            return [42]
    test = Test()
    self.assertIsInstance(test, IFoo)
    self.assertIsInstance(test, IAverage)
    self.assertIsInstance(test, IList)