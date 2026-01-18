import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_parse_mappings_succeeds_for_n_mappings(self):
    self.assertEqual({'key1': 'val1', 'key2': 'val2'}, self.parse(['key1:val1', 'key2:val2']))