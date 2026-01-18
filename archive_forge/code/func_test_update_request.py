import time
import uuid
from designateclient.tests import v2
def test_update_request(self):
    transfer = '098bee04-fe30-4a83-8ccd-e0c496755816'
    project = '098bee04-fe30-4a83-8ccd-e0c496755817'
    ref = {'target_project_id': project}
    parts = ['zones', 'tasks', 'transfer_requests', transfer]
    self.stub_url('PATCH', parts=parts, json=ref)
    self.client.zone_transfers.update_request(transfer, ref)
    self.assertRequestBodyIs(json=ref)