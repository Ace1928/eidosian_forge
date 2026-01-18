from unittest import mock
import testtools
from heatclient.common import hook_utils
import heatclient.v1.shell as shell
def test_update_hooks_in_args(self):
    type(self.args).pre_update = mock.PropertyMock(return_value=['bp', 'another_bp'])
    shell.do_stack_update(self.client, self.args)
    self.assertEqual(1, self.client.stacks.update.call_count)
    expected_hooks = {'bp': {'hooks': 'pre-update'}, 'another_bp': {'hooks': 'pre-update'}}
    actual_hooks = self.client.stacks.update.call_args[1]['environment']['resource_registry']['resources']
    self.assertEqual(expected_hooks, actual_hooks)