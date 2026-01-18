import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_unicode_bool_from_string(self):
    self._test_bool_from_string(str)
    self.assertFalse(strutils.bool_from_string('使用', strict=False))
    exc = self.assertRaises(ValueError, strutils.bool_from_string, '使用', strict=True)
    expected_msg = "Unrecognized value '使用', acceptable values are: '0', '1', 'f', 'false', 'n', 'no', 'off', 'on', 't', 'true', 'y', 'yes'"
    self.assertEqual(expected_msg, str(exc))