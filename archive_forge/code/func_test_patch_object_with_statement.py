from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_patch_object_with_statement(self):

    class Foo(object):
        something = 'foo'
    original = Foo.something
    with patch.object(Foo, 'something'):
        self.assertNotEqual(Foo.something, original, 'unpatched')
    self.assertEqual(Foo.something, original)