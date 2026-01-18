import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
def test_provides_bad(self):
    with self.assertRaises(Exception):

        @provides(Sample)
        class Test(HasTraits):
            pass