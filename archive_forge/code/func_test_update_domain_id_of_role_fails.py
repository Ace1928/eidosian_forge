import uuid
from keystone.common import provider_api
from keystone.common import sql
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.assignment import test_core
from keystone.tests.unit.backend import core_sql
def test_update_domain_id_of_role_fails(self):
    role1 = unit.new_role_ref()
    role1 = PROVIDERS.role_api.create_role(role1['id'], role1)
    domainA = unit.new_domain_ref()
    role1['domain_id'] = domainA['id']
    self.assertRaises(exception.ValidationError, PROVIDERS.role_api.update_role, role1['id'], role1)
    role2 = unit.new_role_ref(domain_id=domainA['id'])
    PROVIDERS.role_api.create_role(role2['id'], role2)
    domainB = unit.new_domain_ref()
    role2['domain_id'] = domainB['id']
    self.assertRaises(exception.ValidationError, PROVIDERS.role_api.update_role, role2['id'], role2)
    role2['domain_id'] = None
    self.assertRaises(exception.ValidationError, PROVIDERS.role_api.update_role, role2['id'], role2)