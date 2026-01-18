import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_spec_set(self):

    class Sub(SomeClass):
        attr = SomeClass()
    for spec in (Sub, Sub()):
        mock = create_autospec(spec, spec_set=True)
        self._check_someclass_mock(mock)
        self.assertRaises(AttributeError, setattr, mock, 'foo', 'bar')
        self.assertRaises(AttributeError, setattr, mock.attr, 'foo', 'bar')