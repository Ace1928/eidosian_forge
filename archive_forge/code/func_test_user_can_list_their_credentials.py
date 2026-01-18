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
def test_user_can_list_their_credentials(self):
    with self.test_client() as c:
        expected = []
        for _ in range(2):
            create = {'credential': {'blob': uuid.uuid4().hex, 'type': uuid.uuid4().hex, 'user_id': self.user_id}}
            r = c.post('/v3/credentials', json=create, headers=self.headers)
            expected.append(r.json['credential'])
        r = c.get('/v3/credentials', headers=self.headers)
        for credential in expected:
            self.assertIn(credential, r.json['credentials'])