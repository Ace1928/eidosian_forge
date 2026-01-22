import os
import unittest
import fixtures
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_exceptions
from openstackclient.tests.functional import base
def _create_dummy_service(self, add_clean_up=True):
    service_name = data_utils.rand_name('TestService')
    description = data_utils.rand_name('description')
    type_name = data_utils.rand_name('TestType')
    raw_output = self.openstack('service create --name %(name)s --description %(description)s %(type)s' % {'name': service_name, 'description': description, 'type': type_name})
    if add_clean_up:
        service = self.parse_show_as_object(raw_output)
        self.addCleanup(self.openstack, 'service delete %s' % service['id'])
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.SERVICE_FIELDS)
    return service_name