import unittest
from traits.api import (
def test_write_only_property_not_copyable(self):
    self.assertNotIn('p_wo', self.names)