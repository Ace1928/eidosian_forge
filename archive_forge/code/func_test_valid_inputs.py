import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
@ddt.unpack
@ddt.data({'value': 42, 'name': 'answer', 'output': 42}, {'value': '42', 'name': 'answer', 'output': 42}, {'value': '7', 'name': 'lucky', 'output': 7, 'min_value': 7, 'max_value': 8}, {'value': 7, 'name': 'lucky', 'output': 7, 'min_value': 6, 'max_value': 7}, {'value': 300, 'name': 'Spartaaa!!!', 'output': 300, 'min_value': 300}, {'value': '300', 'name': 'Spartaaa!!!', 'output': 300, 'max_value': 300})
def test_valid_inputs(self, output, value, name, **kwargs):
    self.assertEqual(strutils.validate_integer(value, name, **kwargs), output)