import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_update_group_name_with_trailing_whitespace(self):
    group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
    group_create = PROVIDERS.identity_api.create_group(group)
    group_name = group['name'] = group['name'] + '    '
    group_update = PROVIDERS.identity_api.update_group(group_create['id'], group)
    self.assertEqual(group_update['id'], group_create['id'])
    self.assertEqual(group_update['name'], group_name.strip())