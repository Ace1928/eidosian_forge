import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as bp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_list_credentials_for_other_users(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user_password = user['password']
    user = PROVIDERS.identity_api.create_user(user)
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project = PROVIDERS.resource_api.create_project(project['id'], project)
    PROVIDERS.assignment_api.create_grant(self.bootstrapper.member_role_id, user_id=user['id'], project_id=project['id'])
    user_auth = self.build_authentication_request(user_id=user['id'], password=user_password, project_id=project['id'])
    with self.test_client() as c:
        r = c.post('/v3/auth/tokens', json=user_auth)
        token_id = r.headers['X-Subject-Token']
        headers = {'X-Auth-Token': token_id}
        create = {'credential': {'blob': uuid.uuid4().hex, 'type': uuid.uuid4().hex, 'user_id': user['id']}}
        r = c.post('/v3/credentials', json=create, headers=headers)
        credential_id = r.json['credential']['id']
    with self.test_client() as c:
        r = c.get('/v3/credentials', headers=self.headers)
        self.assertEqual(1, len(r.json['credentials']))
        self.assertEqual(credential_id, r.json['credentials'][0]['id'])
        self.assertEqual(user['id'], r.json['credentials'][0]['user_id'])