import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_with_backslash_inside_unquoted(self):
    self.check(['a\\bc', 'de'], 'a\\bc,de')