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
def test_override_tuple_methods(self):
    c = call.count()
    i = call.index(132, 'hello')
    m = Mock()
    m.count()
    m.index(132, 'hello')
    self.assertEqual(m.method_calls[0], c)
    self.assertEqual(m.method_calls[1], i)