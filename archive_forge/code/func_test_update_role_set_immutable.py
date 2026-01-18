import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_update_role_set_immutable(self):
    role = unit.new_role_ref()
    role_id = role['id']
    PROVIDERS.role_api.create_role(role_id, role)
    update_role = {'options': {ro_opt.IMMUTABLE_OPT.option_name: True}}
    role_via_manager = PROVIDERS.role_api.get_role(role_id)
    self.assertTrue('options' in role_via_manager)
    self.assertFalse(ro_opt.IMMUTABLE_OPT.option_name in role_via_manager['options'])
    role_update = PROVIDERS.role_api.update_role(role_id, update_role)
    role_via_manager = PROVIDERS.role_api.get_role(role_id)
    self.assertTrue(ro_opt.IMMUTABLE_OPT.option_name in role_update['options'])
    self.assertTrue(role_update['options'][ro_opt.IMMUTABLE_OPT.option_name])
    self.assertTrue(ro_opt.IMMUTABLE_OPT.option_name in role_via_manager['options'])
    self.assertTrue(role_via_manager['options'][ro_opt.IMMUTABLE_OPT.option_name])