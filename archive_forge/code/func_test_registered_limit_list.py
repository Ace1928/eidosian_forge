import os
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_registered_limit_list(self):
    self._create_dummy_registered_limit()
    raw_output = self.openstack('registered limit list')
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, self.REGISTERED_LIMIT_LIST_HEADERS)