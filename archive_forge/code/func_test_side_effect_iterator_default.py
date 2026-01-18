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
def test_side_effect_iterator_default(self):
    mock = Mock(return_value=2)
    mock.side_effect = iter([1, DEFAULT])
    self.assertEqual([mock(), mock()], [1, 2])