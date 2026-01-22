import os
import unittest
import fixtures
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_exceptions
from openstackclient.tests.functional import base
def _create_dummy_endpoint(self, add_clean_up=True):
    region_id = data_utils.rand_name('TestRegion')
    service_name = self._create_dummy_service()
    public_url = data_utils.rand_url()
    admin_url = data_utils.rand_url()
    internal_url = data_utils.rand_url()
    raw_output = self.openstack('endpoint create --publicurl %(publicurl)s --adminurl %(adminurl)s --internalurl %(internalurl)s --region %(region)s %(service)s' % {'publicurl': public_url, 'adminurl': admin_url, 'internalurl': internal_url, 'region': region_id, 'service': service_name})
    endpoint = self.parse_show_as_object(raw_output)
    if add_clean_up:
        self.addCleanup(self.openstack, 'endpoint delete %s' % endpoint['id'])
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.ENDPOINT_FIELDS)
    return endpoint['id']