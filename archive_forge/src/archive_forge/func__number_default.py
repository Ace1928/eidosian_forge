import unittest
from traits.api import (
from traits.observation.api import (
def _number_default(self):
    self.default_call_count += 1
    return 99