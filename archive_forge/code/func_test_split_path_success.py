import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_split_path_success(self):
    self.assertEqual(strutils.split_path('/a'), ['a'])
    self.assertEqual(strutils.split_path('/a/'), ['a'])
    self.assertEqual(strutils.split_path('/a/c', 2), ['a', 'c'])
    self.assertEqual(strutils.split_path('/a/c/o', 3), ['a', 'c', 'o'])
    self.assertEqual(strutils.split_path('/a/c/o/r', 3, 3, True), ['a', 'c', 'o/r'])
    self.assertEqual(strutils.split_path('/a/c', 2, 3, True), ['a', 'c', None])
    self.assertEqual(strutils.split_path('/a/c/', 2), ['a', 'c'])
    self.assertEqual(strutils.split_path('/a/c/', 2, 3), ['a', 'c', ''])