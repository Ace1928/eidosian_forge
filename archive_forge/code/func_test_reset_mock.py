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
def test_reset_mock(self):
    parent = Mock()
    spec = ['something']
    mock = Mock(name='child', parent=parent, spec=spec)
    mock(sentinel.Something, something=sentinel.SomethingElse)
    something = mock.something
    mock.something()
    mock.side_effect = sentinel.SideEffect
    return_value = mock.return_value
    return_value()
    mock.reset_mock()
    self.assertEqual(mock._mock_name, 'child', 'name incorrectly reset')
    self.assertEqual(mock._mock_parent, parent, 'parent incorrectly reset')
    self.assertEqual(mock._mock_methods, spec, 'methods incorrectly reset')
    self.assertFalse(mock.called, 'called not reset')
    self.assertEqual(mock.call_count, 0, 'call_count not reset')
    self.assertEqual(mock.call_args, None, 'call_args not reset')
    self.assertEqual(mock.call_args_list, [], 'call_args_list not reset')
    self.assertEqual(mock.method_calls, [], 'method_calls not initialised correctly: %r != %r' % (mock.method_calls, []))
    self.assertEqual(mock.mock_calls, [])
    self.assertEqual(mock.side_effect, sentinel.SideEffect, 'side_effect incorrectly reset')
    self.assertEqual(mock.return_value, return_value, 'return_value incorrectly reset')
    self.assertFalse(return_value.called, 'return value mock not reset')
    self.assertEqual(mock._mock_children, {'something': something}, 'children reset incorrectly')
    self.assertEqual(mock.something, something, 'children incorrectly cleared')
    self.assertFalse(mock.something.called, 'child not reset')