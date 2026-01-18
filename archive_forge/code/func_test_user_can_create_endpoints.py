import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_create_endpoints(self):
    service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
    create = {'endpoint': {'interface': 'public', 'service_id': service['id'], 'url': 'https://' + uuid.uuid4().hex + '.com'}}
    with self.test_client() as c:
        c.post('/v3/endpoints', json=create, headers=self.headers)