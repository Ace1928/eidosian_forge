import unittest
from traits.api import (
from traits.observation.api import (
@observe('foo')
def _optional_second_argument(self, event=None):
    self.call_count += 1