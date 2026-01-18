import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_show_domain_limit(self):
    ref1 = unit.new_limit_ref(domain_id=self.domain_id, service_id=self.service_id2, resource_name='snapshot')
    r = self.post('/limits', body={'limits': [ref1]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    id1 = r.result['limits'][0]['id']
    r = self.get('/limits/%s' % id1, expected_status=http.client.OK, auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain_id))
    limit = r.result['limit']
    self.assertIsNone(limit['project_id'])
    self.assertIsNone(limit['region_id'])
    for key in ['service_id', 'resource_name', 'resource_limit', 'description', 'domain_id']:
        self.assertEqual(limit[key], ref1[key])