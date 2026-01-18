import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_runtime_error(self):
    f = RaisesArgumentlessRuntimeError()
    self.assertRaises(RuntimeError, setattr, f, 'x', 5)