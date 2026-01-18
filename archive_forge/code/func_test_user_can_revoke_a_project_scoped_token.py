import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_revoke_a_project_scoped_token(self):
    project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user['id'] = PROVIDERS.identity_api.create_user(user)['id']
    PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], project_id=project['id'])
    project_auth = self.build_authentication_request(user_id=user['id'], password=user['password'], project_id=project['id'])
    with self.test_client() as c:
        r = c.post('/v3/auth/tokens', json=project_auth)
        project_token = r.headers['X-Subject-Token']
    with self.test_client() as c:
        self.headers['X-Subject-Token'] = project_token
        c.delete('/v3/auth/tokens', headers=self.headers)