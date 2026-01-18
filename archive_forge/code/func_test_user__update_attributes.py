from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_users
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_user__update_attributes(self, mock_find):
    args = ['userinstance', 'user1', '--host', '1.1.1.1', '--new_name', 'user2', '--new_password', '111111', '--new_host', '1.1.1.2']
    verifylist = [('instance', 'userinstance'), ('name', 'user1'), ('host', '1.1.1.1'), ('new_name', 'user2'), ('new_password', '111111'), ('new_host', '1.1.1.2')]
    mock_find.return_value = args[0]
    parsed_args = self.check_parser(self.cmd, args, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertIsNone(result)