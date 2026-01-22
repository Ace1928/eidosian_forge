import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
class TestMapping(collections.abc.Mapping):
    """Test class for non-dict mappings"""

    def __init__(self):
        super(TestMapping, self).__init__()
        self.data = {'password': 'shhh', 'foo': 'bar'}

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return len(self.data)