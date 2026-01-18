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
def test_setting_attribute_with_spec_set(self):

    class X(object):
        y = 3
    mock = Mock(spec=X)
    mock.x = 'foo'
    mock = Mock(spec_set=X)

    def set_attr():
        mock.x = 'foo'
    mock.y = 'foo'
    self.assertRaises(AttributeError, set_attr)