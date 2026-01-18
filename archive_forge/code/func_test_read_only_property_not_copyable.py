import unittest
from traits.api import (
def test_read_only_property_not_copyable(self):
    self.assertNotIn('p_ro', self.names)