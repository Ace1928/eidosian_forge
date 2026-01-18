from tempest.lib.common.utils import data_utils as utils
from heatclient.tests.functional import config
from heatclient.tests.functional.osc.v1 import base
def test_stack_snapshot_create_restore(self):
    snapshot_name = utils.rand_name(name='test-stack-snapshot')
    stack = self._stack_create_minimal()
    snapshot = self._stack_snapshot_create(stack['id'], snapshot_name)
    self.assertEqual(snapshot_name, snapshot['name'])
    self._stack_snapshot_restore(stack['id'], snapshot['id'])