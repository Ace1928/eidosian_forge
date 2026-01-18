import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_delete_endpoints(self):
    service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
    endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
    endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
    with self.test_client() as c:
        c.delete('/v3/endpoints/%s' % endpoint['id'], headers=self.headers)