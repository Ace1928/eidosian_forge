import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_parse_mappings_succeeds_for_nonuniq_key(self):
    self.assertEqual({'key': ['val1', 'val2']}, self.parse(['key:val1', 'key:val2', 'key:val2'], unique_keys=False))