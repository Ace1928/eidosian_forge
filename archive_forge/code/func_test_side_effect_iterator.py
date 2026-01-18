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
def test_side_effect_iterator(self):
    mock = Mock(side_effect=iter([1, 2, 3]))
    self.assertEqual([mock(), mock(), mock()], [1, 2, 3])
    self.assertRaises(StopIteration, mock)
    mock = MagicMock(side_effect=['a', 'b', 'c'])
    self.assertEqual([mock(), mock(), mock()], ['a', 'b', 'c'])
    self.assertRaises(StopIteration, mock)
    mock = Mock(side_effect='ghi')
    self.assertEqual([mock(), mock(), mock()], ['g', 'h', 'i'])
    self.assertRaises(StopIteration, mock)

    class Foo(object):
        pass
    mock = MagicMock(side_effect=Foo)
    self.assertIsInstance(mock(), Foo)
    mock = Mock(side_effect=Iter())
    self.assertEqual([mock(), mock(), mock(), mock()], ['this', 'is', 'an', 'iter'])
    self.assertRaises(StopIteration, mock)