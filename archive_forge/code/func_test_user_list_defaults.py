from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_users
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_user_list_defaults(self, mock_find):
    args = ['my_instance']
    mock_find.return_value = args[0]
    parsed_args = self.check_parser(self.cmd, args, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.user_client.list.assert_called_once_with(*args)
    self.assertEqual(self.columns, columns)
    self.assertEqual([self.values], data)