import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def _record_trait_list_event(self, object, name, old, new):
    self.last_event = new