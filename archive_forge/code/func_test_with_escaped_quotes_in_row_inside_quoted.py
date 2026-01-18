import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_with_escaped_quotes_in_row_inside_quoted(self):
    self.check(['a"b""c', 'd'], '"a\\"b\\"\\"c",d')