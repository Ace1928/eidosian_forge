import unittest
from traits.api import (
from traits.observation.api import (
@observe('foo')
def _method_with_star_args(self, *args):
    self.call_count += 1