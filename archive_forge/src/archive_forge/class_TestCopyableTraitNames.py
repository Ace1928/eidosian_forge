import unittest
from traits.api import (
class TestCopyableTraitNames(unittest.TestCase):
    """ Validate that copyable_trait_names returns the appropriate result.
    """

    def setUp(self):
        foo = Foo()
        self.names = foo.copyable_trait_names()

    def test_events_not_copyable(self):
        self.assertNotIn('e', self.names)

    def test_read_only_property_not_copyable(self):
        self.assertNotIn('p_ro', self.names)

    def test_write_only_property_not_copyable(self):
        self.assertNotIn('p_wo', self.names)

    def test_any_copyable(self):
        self.assertIn('a', self.names)

    def test_bool_copyable(self):
        self.assertIn('b', self.names)

    def test_str_copyable(self):
        self.assertIn('s', self.names)

    def test_instance_copyable(self):
        self.assertIn('i', self.names)

    def test_delegate_copyable(self):
        self.assertIn('d', self.names)

    def test_property_copyable(self):
        self.assertIn('p', self.names)