import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _DomainAndProjectUserEndpointTests(object):

    def test_user_cannot_create_endpoints(self):
        create = {'endpoint': {'interface': 'public', 'service_id': uuid.uuid4().hex, 'url': 'https://' + uuid.uuid4().hex + '.com'}}
        with self.test_client() as c:
            c.post('/v3/endpoints', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_list_endpoints(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        with self.test_client() as c:
            c.get('/v3/endpoints', headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_get_an_endpoint(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        with self.test_client() as c:
            c.get('/v3/endpoints/%s' % endpoint['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_endpoints(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        update = {'endpoint': {'interface': 'internal'}}
        with self.test_client() as c:
            c.patch('/v3/endpoints/%s' % endpoint['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_endpoints(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        with self.test_client() as c:
            c.delete('/v3/endpoints/%s' % endpoint['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)