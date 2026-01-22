import unittest
from traits.api import (
from traits.observation.api import (
@observe('foo')
def _method_with_extra_optional_args(self, event, frombicate=True):
    self.call_count += 1