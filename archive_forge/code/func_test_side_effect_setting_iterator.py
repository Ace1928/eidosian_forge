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
def test_side_effect_setting_iterator(self):
    mock = Mock()
    mock.side_effect = iter([1, 2, 3])
    self.assertEqual([mock(), mock(), mock()], [1, 2, 3])
    self.assertRaises(StopIteration, mock)
    side_effect = mock.side_effect
    self.assertIsInstance(side_effect, type(iter([])))
    mock.side_effect = ['a', 'b', 'c']
    self.assertEqual([mock(), mock(), mock()], ['a', 'b', 'c'])
    self.assertRaises(StopIteration, mock)
    side_effect = mock.side_effect
    self.assertIsInstance(side_effect, type(iter([])))
    this_iter = Iter()
    mock.side_effect = this_iter
    self.assertEqual([mock(), mock(), mock(), mock()], ['this', 'is', 'an', 'iter'])
    self.assertRaises(StopIteration, mock)
    self.assertIs(mock.side_effect, this_iter)