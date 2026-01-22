import os
import fixtures
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional import base
def _create_dummy_limit(self, add_clean_up=True):
    registered_limit_id = self._create_dummy_registered_limit()
    raw_output = self.openstack('registered limit show %s' % registered_limit_id, cloud=SYSTEM_CLOUD)
    items = self.parse_show(raw_output)
    resource_name = self._extract_value_from_items('resource_name', items)
    service_id = self._extract_value_from_items('service_id', items)
    resource_limit = 15
    project_name = self._create_dummy_project()
    raw_output = self.openstack('project show %s' % project_name)
    items = self.parse_show(raw_output)
    project_id = self._extract_value_from_items('id', items)
    params = {'project_id': project_id, 'service_id': service_id, 'resource_name': resource_name, 'resource_limit': resource_limit}
    raw_output = self.openstack('limit create --project %(project_id)s --service %(service_id)s --resource-limit %(resource_limit)s %(resource_name)s' % params, cloud=SYSTEM_CLOUD)
    items = self.parse_show(raw_output)
    limit_id = self._extract_value_from_items('id', items)
    if add_clean_up:
        self.addCleanup(self.openstack, 'limit delete %s' % limit_id, cloud=SYSTEM_CLOUD)
    self.assert_show_fields(items, self.LIMIT_FIELDS)
    return limit_id