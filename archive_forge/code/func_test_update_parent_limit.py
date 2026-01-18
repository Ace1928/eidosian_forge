import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_parent_limit(self):
    ref_A = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=10)
    ref_B = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=6)
    ref_C = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=7)
    r = self.post('/limits', body={'limits': [ref_A]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    self.post('/limits', body={'limits': [ref_B, ref_C]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    update_dict = {'resource_limit': 8}
    self.patch('/limits/%s' % r.result['limits'][0]['id'], body={'limit': update_dict}, token=self.system_admin_token, expected_status=http.client.OK)