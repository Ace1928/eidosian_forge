import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_list_identity_providers(self):
    expected_idp_ids = []
    idp = PROVIDERS.federation_api.create_idp(uuid.uuid4().hex, unit.new_identity_provider_ref())
    expected_idp_ids.append(idp['id'])
    with self.test_client() as c:
        r = c.get('/v3/OS-FEDERATION/identity_providers', headers=self.headers)
        for idp in r.json['identity_providers']:
            self.assertIn(idp['id'], expected_idp_ids)