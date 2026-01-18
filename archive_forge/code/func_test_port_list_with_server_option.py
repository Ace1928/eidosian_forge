from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
@mock.patch.object(utils, 'find_resource')
def test_port_list_with_server_option(self, mock_find):
    fake_server = compute_fakes.create_one_server()
    mock_find.return_value = fake_server
    arglist = ['--server', 'fake-server-name']
    verifylist = [('server', 'fake-server-name')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.ports.assert_called_once_with(device_id=fake_server.id, fields=LIST_FIELDS_TO_RETRIEVE)
    mock_find.assert_called_once_with(mock.ANY, 'fake-server-name')
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))