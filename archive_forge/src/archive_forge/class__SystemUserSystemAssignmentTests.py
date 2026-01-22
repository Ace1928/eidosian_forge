import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _SystemUserSystemAssignmentTests(object):

    def test_user_can_list_user_system_role_assignments(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_system_grant_for_user(user['id'], self.bootstrapper.member_role_id)
        with self.test_client() as c:
            r = c.get('/v3/system/users/%s/roles' % user['id'], headers=self.headers)
            self.assertEqual(1, len(r.json['roles']))
            self.assertEqual(self.bootstrapper.member_role_id, r.json['roles'][0]['id'])

    def test_user_can_check_user_system_role_assignment(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_system_grant_for_user(user['id'], self.bootstrapper.member_role_id)
        with self.test_client() as c:
            c.get('/v3/system/users/%s/roles/%s' % (user['id'], self.bootstrapper.member_role_id), headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_user_can_list_group_system_role_assignments(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_system_grant_for_group(group['id'], self.bootstrapper.member_role_id)
        with self.test_client() as c:
            r = c.get('/v3/system/groups/%s/roles' % group['id'], headers=self.headers)
            self.assertEqual(1, len(r.json['roles']))
            self.assertEqual(self.bootstrapper.member_role_id, r.json['roles'][0]['id'])

    def test_user_can_check_group_system_role_assignments(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_system_grant_for_group(group['id'], self.bootstrapper.member_role_id)
        with self.test_client() as c:
            c.get('/v3/system/groups/%s/roles/%s' % (group['id'], self.bootstrapper.member_role_id), headers=self.headers, expected_status_code=http.client.NO_CONTENT)