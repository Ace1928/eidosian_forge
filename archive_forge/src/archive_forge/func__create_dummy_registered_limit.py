import os
import fixtures
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional import base
def _create_dummy_registered_limit(self, add_clean_up=True):
    service_name = self._create_dummy_service()
    resource_name = data_utils.rand_name('resource_name')
    params = {'service_name': service_name, 'default_limit': 10, 'resource_name': resource_name}
    raw_output = self.openstack('registered limit create --service %(service_name)s --default-limit %(default_limit)s %(resource_name)s' % params, cloud=SYSTEM_CLOUD)
    items = self.parse_show(raw_output)
    registered_limit_id = self._extract_value_from_items('id', items)
    if add_clean_up:
        self.addCleanup(self.openstack, 'registered limit delete %s' % registered_limit_id, cloud=SYSTEM_CLOUD)
    self.assert_show_fields(items, self.REGISTERED_LIMIT_FIELDS)
    return registered_limit_id