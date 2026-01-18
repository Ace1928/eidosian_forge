import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_builtins_without_create(self):

    @patch(__name__ + '.ord')
    def test_ord(mock_ord):
        mock_ord.return_value = 101
        return ord('c')

    @patch(__name__ + '.open')
    def test_open(mock_open):
        m = mock_open.return_value
        m.read.return_value = 'abcd'
        fobj = open('doesnotexists.txt')
        data = fobj.read()
        fobj.close()
        return data
    self.assertEqual(test_ord(), 101)
    self.assertEqual(test_open(), 'abcd')