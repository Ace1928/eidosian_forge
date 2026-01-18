import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_strict_bool_from_string(self):
    exc = self.assertRaises(ValueError, strutils.bool_from_string, None, strict=True)
    expected_msg = "Unrecognized value 'None', acceptable values are: '0', '1', 'f', 'false', 'n', 'no', 'off', 'on', 't', 'true', 'y', 'yes'"
    self.assertEqual(expected_msg, str(exc))
    self.assertFalse(strutils.bool_from_string('Other', strict=False))
    exc = self.assertRaises(ValueError, strutils.bool_from_string, 'Other', strict=True)
    expected_msg = "Unrecognized value 'Other', acceptable values are: '0', '1', 'f', 'false', 'n', 'no', 'off', 'on', 't', 'true', 'y', 'yes'"
    self.assertEqual(expected_msg, str(exc))
    exc = self.assertRaises(ValueError, strutils.bool_from_string, 2, strict=True)
    expected_msg = "Unrecognized value '2', acceptable values are: '0', '1', 'f', 'false', 'n', 'no', 'off', 'on', 't', 'true', 'y', 'yes'"
    self.assertEqual(expected_msg, str(exc))
    self.assertFalse(strutils.bool_from_string('f', strict=True))
    self.assertFalse(strutils.bool_from_string('false', strict=True))
    self.assertFalse(strutils.bool_from_string('off', strict=True))
    self.assertFalse(strutils.bool_from_string('n', strict=True))
    self.assertFalse(strutils.bool_from_string('no', strict=True))
    self.assertFalse(strutils.bool_from_string('0', strict=True))
    self.assertTrue(strutils.bool_from_string('1', strict=True))
    for char in ('O', 'o', 'L', 'l', 'I', 'i'):
        self.assertRaises(ValueError, strutils.bool_from_string, char, strict=True)