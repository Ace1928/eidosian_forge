import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_split_path_invalid_path(self):
    try:
        strutils.split_path('o\nn e', 2)
    except ValueError as err:
        self.assertEqual(str(err), 'Invalid path: o%0An%20e')
    try:
        strutils.split_path('o\nn e', 2, 3, True)
    except ValueError as err:
        self.assertEqual(str(err), 'Invalid path: o%0An%20e')