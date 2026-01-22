import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
class StringLengthTestCase(test_base.BaseTestCase):

    def test_check_string_length(self):
        self.assertIsNone(strutils.check_string_length('test', 'name', max_length=255))
        self.assertRaises(ValueError, strutils.check_string_length, '', 'name', min_length=1)
        self.assertRaises(ValueError, strutils.check_string_length, 'a' * 256, 'name', max_length=255)
        self.assertRaises(TypeError, strutils.check_string_length, 11, 'name', max_length=255)
        self.assertRaises(TypeError, strutils.check_string_length, dict(), 'name', max_length=255)

    def test_check_string_length_noname(self):
        self.assertIsNone(strutils.check_string_length('test', max_length=255))
        self.assertRaises(ValueError, strutils.check_string_length, '', min_length=1)
        self.assertRaises(ValueError, strutils.check_string_length, 'a' * 256, max_length=255)
        self.assertRaises(TypeError, strutils.check_string_length, 11, max_length=255)
        self.assertRaises(TypeError, strutils.check_string_length, dict(), max_length=255)