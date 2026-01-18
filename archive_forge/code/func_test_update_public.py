from unittest import mock
from mistralclient.api.v2 import actions
from mistralclient.commands.v2 import actions as action_cmd
from mistralclient.commands.v2 import base as cmd_base
from mistralclient.tests.unit import base
@mock.patch('argparse.open', create=True)
def test_update_public(self, mock_open):
    self.client.actions.update.return_value = [ACTION]
    result = self.call(action_cmd.Update, app_args=['my_action.yaml', '--public'])
    self.assertEqual([('1234-4567-7894-7895', 'a', True, 'param1', 'My cool action', 'test', '1', '1')], result[1])
    self.assertEqual('public', self.client.actions.update.call_args[1]['scope'])