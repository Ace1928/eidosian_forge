import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_get_an_identity_provider(self):
    idp = PROVIDERS.federation_api.create_idp(uuid.uuid4().hex, unit.new_identity_provider_ref())
    with self.test_client() as c:
        c.get('/v3/OS-FEDERATION/identity_providers/%s' % idp['id'], headers=self.headers)