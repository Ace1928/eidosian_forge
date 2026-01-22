import os
import fixtures
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional import base
def _create_dummy_sp(self, add_clean_up=True):
    service_provider = data_utils.rand_name('ServiceProvider')
    description = data_utils.rand_name('description')
    raw_output = self.openstack('service provider create  %(name)s --description %(description)s --auth-url https://sp.example.com:35357 --service-provider-url https://sp.example.com:5000 --enable ' % {'name': service_provider, 'description': description})
    if add_clean_up:
        self.addCleanup(self.openstack, 'service provider delete %s' % service_provider)
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.SERVICE_PROVIDER_FIELDS)
    return service_provider