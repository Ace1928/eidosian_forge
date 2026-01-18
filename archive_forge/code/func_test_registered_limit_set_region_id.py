import os
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_registered_limit_set_region_id(self):
    region_id = self._create_dummy_region()
    registered_limit_id = self._create_dummy_registered_limit()
    params = {'registered_limit_id': registered_limit_id, 'region_id': region_id}
    raw_output = self.openstack('registered limit set %(registered_limit_id)s --region %(region_id)s' % params, cloud=SYSTEM_CLOUD)
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.REGISTERED_LIMIT_FIELDS)