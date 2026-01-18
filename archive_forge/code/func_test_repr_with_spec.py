import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_repr_with_spec(self):

    class X(object):
        pass
    mock = Mock(spec=X)
    self.assertIn(" spec='X' ", repr(mock))
    mock = Mock(spec=X())
    self.assertIn(" spec='X' ", repr(mock))
    mock = Mock(spec_set=X)
    self.assertIn(" spec_set='X' ", repr(mock))
    mock = Mock(spec_set=X())
    self.assertIn(" spec_set='X' ", repr(mock))
    mock = Mock(spec=X, name='foo')
    self.assertIn(" spec='X' ", repr(mock))
    self.assertIn(" name='foo' ", repr(mock))
    mock = Mock(name='foo')
    self.assertNotIn('spec', repr(mock))
    mock = Mock()
    self.assertNotIn('spec', repr(mock))
    mock = Mock(spec=['foo'])
    self.assertNotIn('spec', repr(mock))