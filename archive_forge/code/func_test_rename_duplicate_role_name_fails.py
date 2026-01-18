import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_rename_duplicate_role_name_fails(self):
    role_id1 = uuid.uuid4().hex
    role_id2 = uuid.uuid4().hex
    role1 = unit.new_role_ref(id=role_id1, name='fake1name')
    role2 = unit.new_role_ref(id=role_id2, name='fake2name')
    PROVIDERS.role_api.create_role(role_id1, role1)
    PROVIDERS.role_api.create_role(role_id2, role2)
    role1['name'] = 'fake2name'
    self.assertRaises(exception.Conflict, PROVIDERS.role_api.update_role, role_id1, role1)