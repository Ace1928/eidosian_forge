from tempest.lib.common.utils import data_utils as utils
from heatclient.tests.functional import config
from heatclient.tests.functional.osc.v1 import base
def test_stack_create_minimal_from_url(self):
    stack = self._stack_create_minimal(from_url=True)
    self.assertEqual(self.stack_name, stack['stack_name'])
    self.assertEqual('CREATE_COMPLETE', stack['stack_status'])