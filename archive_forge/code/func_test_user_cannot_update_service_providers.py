import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_update_service_providers(self):
    service_provider = PROVIDERS.federation_api.create_sp(uuid.uuid4().hex, unit.new_service_provider_ref())
    update = {'service_provider': {'enabled': False}}
    with self.test_client() as c:
        c.patch('/v3/OS-FEDERATION/service_providers/%s' % service_provider['id'], headers=self.headers, json=update, expected_status_code=http.client.FORBIDDEN)