import json
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _DomainAndProjectUserPolicyTests(object):

    def test_user_cannot_list_policies(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        with self.test_client() as c:
            c.get('/v3/policies', headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_get_policy(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        with self.test_client() as c:
            c.get('/v3/policies/%s' % policy['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_create_policy(self):
        create = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'description': uuid.uuid4().hex, 'enabled': True, 'blob': json.dumps({'data': uuid.uuid4().hex}), 'type': uuid.uuid4().hex}
        with self.test_client() as c:
            c.post('/v3/policies', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_policy(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        update = {'policy': {'name': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/policies/%s' % policy['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_policy(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        with self.test_client() as c:
            c.delete('/v3/policies/%s' % policy['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)