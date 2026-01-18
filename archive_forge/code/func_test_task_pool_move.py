import time
import uuid
from designateclient.tests import v2
def test_task_pool_move(self):
    ref = self.new_ref(pool_id=1)
    parts = [self.RESOURCE, ref['id'], 'tasks', 'pool_move']
    self.stub_url('POST', parts=parts)
    values = ref.copy()
    self.client.zones.pool_move(ref['id'], values)
    self.assertRequestBodyIs(json=values)