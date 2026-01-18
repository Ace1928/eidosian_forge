import time
import uuid
from designateclient.tests import v2
def test_share_a_zone(self):
    json_body = {'target_project_id': self.target_project_id}
    expected = self.new_ref()
    self.stub_entity('POST', parts=['zones', self.zone_id, 'shares'], entity=expected, json=json_body)
    response = self.client.zone_share.create(self.zone_id, self.target_project_id)
    self.assertRequestBodyIs(json=json_body)
    self.assertEqual(expected, response)