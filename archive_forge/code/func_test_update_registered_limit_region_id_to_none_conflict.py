import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_registered_limit_region_id_to_none_conflict(self):
    ref1 = unit.new_registered_limit_ref(service_id=self.service_id, resource_name='volume', default_limit=10)
    ref2 = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume', default_limit=10)
    self.post('/registered_limits', body={'registered_limits': [ref1]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    r = self.post('/registered_limits', body={'registered_limits': [ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    update_ref = {'region_id': None}
    registered_limit_id = r.result['registered_limits'][0]['id']
    self.patch('/registered_limits/%s' % registered_limit_id, body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.CONFLICT)