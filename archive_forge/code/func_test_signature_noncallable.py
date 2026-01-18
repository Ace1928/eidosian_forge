import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_signature_noncallable(self):

    class NonCallable(object):

        def __init__(self):
            pass
    mock = create_autospec(NonCallable)
    instance = mock()
    mock.assert_called_once_with()
    self.assertRaises(TypeError, mock, 'a')
    self.assertRaises(TypeError, instance)
    self.assertRaises(TypeError, instance, 'a')
    mock = create_autospec(NonCallable())
    self.assertRaises(TypeError, mock)
    self.assertRaises(TypeError, mock, 'a')