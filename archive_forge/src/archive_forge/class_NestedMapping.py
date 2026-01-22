import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
class NestedMapping(TestMapping):
    """Test class that contains an instance of TestMapping"""

    def __init__(self):
        super(NestedMapping, self).__init__()
        self.data = {'nested': TestMapping()}