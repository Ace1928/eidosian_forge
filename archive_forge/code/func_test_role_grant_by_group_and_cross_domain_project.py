from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_role_grant_by_group_and_cross_domain_project(self):
    role1 = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role1['id'], role1)
    role2 = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role2['id'], role2)
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    domain2 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
    group1 = unit.new_group_ref(domain_id=domain1['id'])
    group1 = PROVIDERS.identity_api.create_group(group1)
    project1 = unit.new_project_ref(domain_id=domain2['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], project_id=project1['id'])
    self.assertEqual(0, len(roles_ref))
    PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=role1['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=role2['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], project_id=project1['id'])
    roles_ref_ids = []
    for ref in roles_ref:
        roles_ref_ids.append(ref['id'])
    self.assertIn(role1['id'], roles_ref_ids)
    self.assertIn(role2['id'], roles_ref_ids)
    PROVIDERS.assignment_api.delete_grant(group_id=group1['id'], project_id=project1['id'], role_id=role1['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], project_id=project1['id'])
    self.assertEqual(1, len(roles_ref))
    self.assertDictEqual(role2, roles_ref[0])