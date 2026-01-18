import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_registered_limit_with_invalid_input(self):
    ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume', default_limit=10)
    r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    reg_id = r.result['registered_limits'][0]['id']
    update_ref1 = unit.new_registered_limit_ref(service_id='fake_id')
    update_ref2 = unit.new_registered_limit_ref(default_limit='not_int')
    update_ref3 = unit.new_registered_limit_ref(resource_name=123)
    update_ref4 = unit.new_registered_limit_ref(region_id='fake_region')
    update_ref5 = unit.new_registered_limit_ref(description=123)
    for input_limit in [update_ref1, update_ref2, update_ref3, update_ref4, update_ref5]:
        self.patch('/registered_limits/%s' % reg_id, body={'registered_limit': input_limit}, token=self.system_admin_token, expected_status=http.client.BAD_REQUEST)