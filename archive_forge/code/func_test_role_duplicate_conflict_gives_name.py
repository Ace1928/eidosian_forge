import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_role_duplicate_conflict_gives_name(self):
    role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role['id'], role)
    role['id'] = uuid.uuid4().hex
    try:
        PROVIDERS.role_api.create_role(role['id'], role)
    except exception.Conflict as e:
        self.assertIn('Duplicate entry found with name %s' % role['name'], repr(e))
    else:
        self.fail('Create duplicate role did not raise a conflict')