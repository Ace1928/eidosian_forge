import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_show_registered_limit(self):
    ref1 = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id)
    ref2 = unit.new_registered_limit_ref(service_id=self.service_id2, region_id=self.region_id2)
    r = self.post('/registered_limits', body={'registered_limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    id1 = r.result['registered_limits'][0]['id']
    self.get('/registered_limits/fake_id', expected_status=http.client.NOT_FOUND)
    r = self.get('/registered_limits/%s' % id1, expected_status=http.client.OK)
    registered_limit = r.result['registered_limit']
    for key in ['service_id', 'region_id', 'resource_name', 'default_limit', 'description']:
        self.assertEqual(registered_limit[key], ref1[key])