import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_safe_sort_key(self):
    data1 = {'k1': 'v1', 'k2': 'v2'}
    data2 = {'k2': 'v2', 'k1': 'v1'}
    self.assertEqual(helpers.safe_sort_key(data1), helpers.safe_sort_key(data2))