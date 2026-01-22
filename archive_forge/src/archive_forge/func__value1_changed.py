import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def _value1_changed(self, old, new):
    self.value1_count += 1