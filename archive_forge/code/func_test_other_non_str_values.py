import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_other_non_str_values(self):
    payload = {'password': 'DK0PK1AK3', 'bool': True, 'dict': {'cat': 'meow', 'password': '*aa38skdjf'}, 'float': 0.1, 'int': 123, 'list': [1, 2], 'none': None, 'str': 'foo'}
    expected = {'password': '***', 'bool': True, 'dict': {'cat': 'meow', 'password': '***'}, 'float': 0.1, 'int': 123, 'list': [1, 2], 'none': None, 'str': 'foo'}
    self.assertEqual(expected, strutils.mask_dict_password(payload))