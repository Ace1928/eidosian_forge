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
def test_side_effect(self):
    mock = Mock()

    def effect(*args, **kwargs):
        raise SystemError('kablooie')
    mock.side_effect = effect
    self.assertRaises(SystemError, mock, 1, 2, fish=3)
    mock.assert_called_with(1, 2, fish=3)
    results = [1, 2, 3]

    def effect():
        return results.pop()
    mock.side_effect = effect
    self.assertEqual([mock(), mock(), mock()], [3, 2, 1], 'side effect not used correctly')
    mock = Mock(side_effect=sentinel.SideEffect)
    self.assertEqual(mock.side_effect, sentinel.SideEffect, 'side effect in constructor not used')

    def side_effect():
        return DEFAULT
    mock = Mock(side_effect=side_effect, return_value=sentinel.RETURN)
    self.assertEqual(mock(), sentinel.RETURN)