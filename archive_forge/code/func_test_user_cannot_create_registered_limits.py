import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_create_registered_limits(self):
    service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
    create = {'registered_limits': [unit.new_registered_limit_ref(service_id=service['id'])]}
    with self.test_client() as c:
        c.post('/v3/registered_limits', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)