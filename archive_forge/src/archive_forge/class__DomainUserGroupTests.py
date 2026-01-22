import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import group as gp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _DomainUserGroupTests(object):

    def test_user_can_list_groups_in_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        group1 = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        group2 = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        with self.test_client() as c:
            r = c.get('/v3/groups', headers=self.headers)
            self.assertEqual(1, len(r.json['groups']))
            self.assertNotIn(group1['id'], [g['id'] for g in r.json['groups']])
            self.assertEqual(group2['id'], r.json['groups'][0]['id'])

    def test_user_cannot_list_groups_in_other_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        with self.test_client() as c:
            r = c.get('/v3/groups?domain_id=%s' % domain['id'], headers=self.headers)
            self.assertEqual(0, len(r.json['groups']))

    def test_user_can_get_group_in_domain(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        with self.test_client() as c:
            r = c.get('/v3/groups/%s' % group['id'], headers=self.headers)
            self.assertEqual(group['id'], r.json['group']['id'])

    def test_user_cannot_get_group_in_other_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        with self.test_client() as c:
            c.get('/v3/groups/%s' % group['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_get_non_existent_group_forbidden(self):
        with self.test_client() as c:
            c.get('/v3/groups/%s' % uuid.uuid4().hex, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_can_list_groups_in_domain_for_user_in_domain(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        with self.test_client() as c:
            r = c.get('/v3/users/%s/groups' % user['id'], headers=self.headers)
            self.assertEqual(1, len(r.json['groups']))
            self.assertEqual(group['id'], r.json['groups'][0]['id'])

    def test_user_cannot_list_groups_in_own_domain_user_in_other_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        with self.test_client() as c:
            c.get('/v3/users/%s/groups' % user['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_list_groups_for_non_existent_user_forbidden(self):
        with self.test_client() as c:
            c.get('/v3/users/%s/groups' % uuid.uuid4().hex, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_list_groups_in_other_domain_user_in_own_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        group1 = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        group2 = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        PROVIDERS.identity_api.add_user_to_group(user['id'], group1['id'])
        PROVIDERS.identity_api.add_user_to_group(user['id'], group2['id'])
        with self.test_client() as c:
            r = c.get('/v3/users/%s/groups' % user['id'], headers=self.headers)
            self.assertEqual(1, len(r.json['groups']))
            self.assertEqual(group2['id'], r.json['groups'][0]['id'])

    def test_user_can_list_users_in_own_domain_for_group_in_own_domain(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        with self.test_client() as c:
            r = c.get('/v3/groups/%s/users' % group['id'], headers=self.headers)
            self.assertEqual(1, len(r.json['users']))
            self.assertEqual(user['id'], r.json['users'][0]['id'])

    def test_user_cannot_list_users_in_other_domain_group_in_own_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        user1 = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
        user2 = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group['id'])
        PROVIDERS.identity_api.add_user_to_group(user2['id'], group['id'])
        with self.test_client() as c:
            r = c.get('/v3/groups/%s/users' % group['id'], headers=self.headers)
            self.assertEqual(1, len(r.json['users']))
            self.assertEqual(user2['id'], r.json['users'][0]['id'])

    def test_user_cannot_list_users_in_own_domain_group_in_other_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        with self.test_client() as c:
            c.get('/v3/groups/%s/users' % group['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_list_users_in_non_existent_group_forbidden(self):
        with self.test_client() as c:
            c.get('/v3/groups/%s/users' % uuid.uuid4().hex, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_can_check_user_in_own_domain_group_in_own_domain(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        with self.test_client() as c:
            c.head('/v3/groups/%(group)s/users/%(user)s' % {'group': group['id'], 'user': user['id']}, headers=self.headers, expected_status_code=http.client.NO_CONTENT)
            c.get('/v3/groups/%(group)s/users/%(user)s' % {'group': group['id'], 'user': user['id']}, headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_user_cannot_check_user_in_other_domain_group_in_own_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        with self.test_client() as c:
            c.head('/v3/groups/%(group)s/users/%(user)s' % {'group': group['id'], 'user': user['id']}, headers=self.headers, expected_status_code=http.client.FORBIDDEN)
            c.get('/v3/groups/%(group)s/users/%(user)s' % {'group': group['id'], 'user': user['id']}, headers=self.headers, expected_status_code=http.client.FORBIDDEN)