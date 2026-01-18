from tempest.lib.common.utils import data_utils as utils
from heatclient.tests.functional import config
from heatclient.tests.functional.osc.v1 import base
def test_stack_suspend_resume(self):
    stack = self._stack_create_minimal()
    stack = self._stack_suspend(stack['id'])
    self.assertEqual(self.stack_name, stack['stack_name'])
    self.assertEqual('SUSPEND_COMPLETE', stack['stack_status'])
    stack = self._stack_resume(stack['id'])
    self.assertEqual(self.stack_name, stack['stack_name'])
    self.assertEqual('RESUME_COMPLETE', stack['stack_status'])