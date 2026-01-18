import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_create_identity_providers(self):
    create = {'identity_provider': {'remote_ids': [uuid.uuid4().hex]}}
    with self.test_client() as c:
        c.put('/v3/OS-FEDERATION/identity_providers/%s' % uuid.uuid4().hex, json=create, headers=self.headers, expected_status_code=http.client.CREATED)