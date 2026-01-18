import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_application_credential(self):
    with self.test_client() as c:
        roles = [{'id': self.role_id}]
        app_cred_body = self._app_cred_body(roles=roles)
        token = self.get_scoped_token()
        resp = c.post('/v3/users/%s/application_credentials' % self.user_id, json=app_cred_body, expected_status_code=http.client.CREATED, headers={'X-Auth-Token': token})
        app_cred_body['application_credential']['description'] = 'New Things'
        app_cred_id = resp.json['application_credential']['id']
        member_path = '/v3%s' % MEMBER_PATH_FMT % {'user_id': self.user_id, 'app_cred_id': app_cred_id}
        c.patch(member_path, json=app_cred_body, expected_status_code=http.client.METHOD_NOT_ALLOWED, headers={'X-Auth-Token': token})