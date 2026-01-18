import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_service_association_cleanup_when_policy_deleted(self):
    url = '/policies/%(policy_id)s/OS-ENDPOINT-POLICY/services/%(service_id)s' % {'policy_id': self.policy['id'], 'service_id': self.service['id']}
    self.put(url)
    self.get(url, expected_status=http.client.NO_CONTENT)
    self.delete('/services/%(service_id)s' % {'service_id': self.service['id']})
    self.head(url, expected_status=http.client.NOT_FOUND)