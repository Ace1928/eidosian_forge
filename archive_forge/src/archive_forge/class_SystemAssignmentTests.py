from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
class SystemAssignmentTests(AssignmentTestHelperMixin):

    def test_create_system_grant_for_user(self):
        user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_id = PROVIDERS.identity_api.create_user(user_ref)['id']
        role_ref = self._create_role()
        PROVIDERS.assignment_api.create_system_grant_for_user(user_id, role_ref['id'])
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_user(user_id)
        self.assertEqual(len(system_roles), 1)
        self.assertIsNone(system_roles[0]['domain_id'])
        self.assertEqual(system_roles[0]['id'], role_ref['id'])
        self.assertEqual(system_roles[0]['name'], role_ref['name'])

    def test_list_system_grants_for_user(self):
        user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_id = PROVIDERS.identity_api.create_user(user_ref)['id']
        first_role = self._create_role()
        second_role = self._create_role()
        PROVIDERS.assignment_api.create_system_grant_for_user(user_id, first_role['id'])
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_user(user_id)
        self.assertEqual(len(system_roles), 1)
        PROVIDERS.assignment_api.create_system_grant_for_user(user_id, second_role['id'])
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_user(user_id)
        self.assertEqual(len(system_roles), 2)

    def test_check_system_grant_for_user(self):
        user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_id = PROVIDERS.identity_api.create_user(user_ref)['id']
        role = self._create_role()
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_system_grant_for_user, user_id, role['id'])
        PROVIDERS.assignment_api.create_system_grant_for_user(user_id, role['id'])
        PROVIDERS.assignment_api.check_system_grant_for_user(user_id, role['id'])

    def test_delete_system_grant_for_user(self):
        user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_id = PROVIDERS.identity_api.create_user(user_ref)['id']
        role = self._create_role()
        PROVIDERS.assignment_api.create_system_grant_for_user(user_id, role['id'])
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_user(user_id)
        self.assertEqual(len(system_roles), 1)
        PROVIDERS.assignment_api.delete_system_grant_for_user(user_id, role['id'])
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_user(user_id)
        self.assertEqual(len(system_roles), 0)

    def test_check_system_grant_for_user_with_invalid_role_fails(self):
        user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_id = PROVIDERS.identity_api.create_user(user_ref)['id']
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_system_grant_for_user, user_id, uuid.uuid4().hex)

    def test_check_system_grant_for_user_with_invalid_user_fails(self):
        role = self._create_role()
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_system_grant_for_user, uuid.uuid4().hex, role['id'])

    def test_delete_system_grant_for_user_with_invalid_role_fails(self):
        user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_id = PROVIDERS.identity_api.create_user(user_ref)['id']
        role = self._create_role()
        PROVIDERS.assignment_api.create_system_grant_for_user(user_id, role['id'])
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_system_grant_for_user, user_id, uuid.uuid4().hex)

    def test_delete_system_grant_for_user_with_invalid_user_fails(self):
        user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_id = PROVIDERS.identity_api.create_user(user_ref)['id']
        role = self._create_role()
        PROVIDERS.assignment_api.create_system_grant_for_user(user_id, role['id'])
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_system_grant_for_user, uuid.uuid4().hex, role['id'])

    def test_list_system_grants_for_user_returns_empty_list(self):
        user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_id = PROVIDERS.identity_api.create_user(user_ref)['id']
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_user(user_id)
        self.assertFalse(system_roles)

    def test_create_system_grant_for_user_fails_with_domain_role(self):
        user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_id = PROVIDERS.identity_api.create_user(user_ref)['id']
        role = self._create_role(domain_id=CONF.identity.default_domain_id)
        self.assertRaises(exception.ValidationError, PROVIDERS.assignment_api.create_system_grant_for_user, user_id, role['id'])

    def test_create_system_grant_for_group(self):
        group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
        group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
        role_ref = self._create_role()
        PROVIDERS.assignment_api.create_system_grant_for_group(group_id, role_ref['id'])
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_group(group_id)
        self.assertEqual(len(system_roles), 1)
        self.assertIsNone(system_roles[0]['domain_id'])
        self.assertEqual(system_roles[0]['id'], role_ref['id'])
        self.assertEqual(system_roles[0]['name'], role_ref['name'])

    def test_list_system_grants_for_group(self):
        group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
        group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
        first_role = self._create_role()
        second_role = self._create_role()
        PROVIDERS.assignment_api.create_system_grant_for_group(group_id, first_role['id'])
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_group(group_id)
        self.assertEqual(len(system_roles), 1)
        PROVIDERS.assignment_api.create_system_grant_for_group(group_id, second_role['id'])
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_group(group_id)
        self.assertEqual(len(system_roles), 2)

    def test_check_system_grant_for_group(self):
        group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
        group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
        role = self._create_role()
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_system_grant_for_group, group_id, role['id'])
        PROVIDERS.assignment_api.create_system_grant_for_group(group_id, role['id'])
        PROVIDERS.assignment_api.check_system_grant_for_group(group_id, role['id'])

    def test_delete_system_grant_for_group(self):
        group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
        group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
        role = self._create_role()
        PROVIDERS.assignment_api.create_system_grant_for_group(group_id, role['id'])
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_group(group_id)
        self.assertEqual(len(system_roles), 1)
        PROVIDERS.assignment_api.delete_system_grant_for_group(group_id, role['id'])
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_group(group_id)
        self.assertEqual(len(system_roles), 0)

    def test_check_system_grant_for_group_with_invalid_role_fails(self):
        group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
        group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_system_grant_for_group, group_id, uuid.uuid4().hex)

    def test_check_system_grant_for_group_with_invalid_group_fails(self):
        role = self._create_role()
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_system_grant_for_group, uuid.uuid4().hex, role['id'])

    def test_delete_system_grant_for_group_with_invalid_role_fails(self):
        group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
        group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
        role = self._create_role()
        PROVIDERS.assignment_api.create_system_grant_for_group(group_id, role['id'])
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_system_grant_for_group, group_id, uuid.uuid4().hex)

    def test_delete_system_grant_for_group_with_invalid_group_fails(self):
        group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
        group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
        role = self._create_role()
        PROVIDERS.assignment_api.create_system_grant_for_group(group_id, role['id'])
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_system_grant_for_group, uuid.uuid4().hex, role['id'])

    def test_list_system_grants_for_group_returns_empty_list(self):
        group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
        group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_group(group_id)
        self.assertFalse(system_roles)

    def test_create_system_grant_for_group_fails_with_domain_role(self):
        group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
        group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
        role = self._create_role(CONF.identity.default_domain_id)
        self.assertRaises(exception.ValidationError, PROVIDERS.assignment_api.create_system_grant_for_group, group_id, role['id'])

    def test_delete_role_with_system_assignments(self):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user = unit.new_user_ref(domain_id=domain['id'])
        user = PROVIDERS.identity_api.create_user(user)
        PROVIDERS.assignment_api.create_system_grant_for_user(user['id'], role['id'])
        PROVIDERS.role_api.delete_role(role['id'])
        system_roles = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
        self.assertEqual(len(system_roles), 0)