import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_spec_as_list(self):
    mock = create_autospec([])
    mock.append('foo')
    mock.append.assert_called_with('foo')
    self.assertRaises(AttributeError, getattr, mock, 'foo')

    class Foo(object):
        foo = []
    mock = create_autospec(Foo)
    mock.foo.append(3)
    mock.foo.append.assert_called_with(3)
    self.assertRaises(AttributeError, getattr, mock.foo, 'foo')