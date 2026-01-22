import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def _value2_changed(self, old, new):
    self.value2_count += 1