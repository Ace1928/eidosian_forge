from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_sp_list(self):
    self._create_dummy_sp(add_clean_up=True)
    raw_output = self.openstack('service provider list')
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, self.SERVICE_PROVIDER_LIST_HEADERS)