import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_list_services(self):
    expected_service_ids = []
    for _ in range(2):
        s = unit.new_service_ref()
        service = PROVIDERS.catalog_api.create_service(s['id'], s)
        expected_service_ids.append(service['id'])
    with self.test_client() as c:
        r = c.get('/v3/services', headers=self.headers)
        actual_service_ids = []
        for service in r.json['services']:
            actual_service_ids.append(service['id'])
        for service_id in expected_service_ids:
            self.assertIn(service_id, actual_service_ids)