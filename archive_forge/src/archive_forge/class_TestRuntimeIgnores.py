import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
class TestRuntimeIgnores(TestCase):

    def setUp(self):
        super().setUp()
        self.overrideAttr(ignores, '_runtime_ignores', set())

    def test_add(self):
        """Test that we can add an entry to the list."""
        self.assertEqual(set(), ignores.get_runtime_ignores())
        ignores.add_runtime_ignores(['foo'])
        self.assertEqual({'foo'}, ignores.get_runtime_ignores())

    def test_add_duplicate(self):
        """Adding the same ignore twice shouldn't add a new entry."""
        ignores.add_runtime_ignores(['foo', 'bar'])
        self.assertEqual({'foo', 'bar'}, ignores.get_runtime_ignores())
        ignores.add_runtime_ignores(['bar'])
        self.assertEqual({'foo', 'bar'}, ignores.get_runtime_ignores())