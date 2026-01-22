import os
import fixtures
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional import base
def _create_dummy_group(self, add_clean_up=True):
    group_name = data_utils.rand_name('TestGroup')
    description = data_utils.rand_name('description')
    raw_output = self.openstack('group create --domain %(domain)s --description %(description)s %(name)s' % {'domain': self.domain_name, 'description': description, 'name': group_name})
    if add_clean_up:
        self.addCleanup(self.openstack, 'group delete --domain %(domain)s %(name)s' % {'domain': self.domain_name, 'name': group_name})
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.GROUP_FIELDS)
    return group_name