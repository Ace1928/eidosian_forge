import time
import uuid
from designateclient.tests import v2
def test_task_axfr(self):
    ref = self.new_ref()
    parts = [self.RESOURCE, ref['id'], 'tasks', 'xfr']
    self.stub_url('POST', parts=parts)
    self.client.zones.axfr(ref['id'])
    self.assertRequestBodyIs(None)