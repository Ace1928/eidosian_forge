import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_registered_limit_with_referenced_limit(self):
    ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume', default_limit=10)
    r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    update_ref = {'service_id': self.service_id2, 'region_id': self.region_id2, 'resource_name': 'snapshot', 'default_limit': 5}
    self.patch('/registered_limits/%s' % r.result['registered_limits'][0]['id'], body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)