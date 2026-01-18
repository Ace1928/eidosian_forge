import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_revoke_a_system_scoped_token(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user['id'] = PROVIDERS.identity_api.create_user(user)['id']
    PROVIDERS.assignment_api.create_system_grant_for_user(user['id'], self.bootstrapper.reader_role_id)
    system_auth = self.build_authentication_request(user_id=user['id'], password=user['password'], system=True)
    with self.test_client() as c:
        r = c.post('/v3/auth/tokens', json=system_auth)
        system_token = r.headers['X-Subject-Token']
    with self.test_client() as c:
        self.headers['X-Subject-Token'] = system_token
        c.delete('/v3/auth/tokens', headers=self.headers, expected_status_code=http.client.FORBIDDEN)