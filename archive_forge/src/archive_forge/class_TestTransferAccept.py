from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_transfer_request
class TestTransferAccept(TestTransfer):
    columns = ('id', 'name', 'volume_id')

    def setUp(self):
        super().setUp()
        self.volume_transfer = volume_fakes.create_one_transfer()
        self.data = (self.volume_transfer.id, self.volume_transfer.name, self.volume_transfer.volume_id)
        self.transfer_mock.get.return_value = self.volume_transfer
        self.transfer_mock.accept.return_value = self.volume_transfer
        self.cmd = volume_transfer_request.AcceptTransferRequest(self.app, None)

    def test_transfer_accept(self):
        arglist = ['--auth-key', 'key_value', self.volume_transfer.id]
        verifylist = [('transfer_request', self.volume_transfer.id), ('auth_key', 'key_value')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.transfer_mock.get.assert_called_once_with(self.volume_transfer.id)
        self.transfer_mock.accept.assert_called_once_with(self.volume_transfer.id, 'key_value')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_transfer_accept_no_option(self):
        arglist = [self.volume_transfer.id]
        verifylist = [('transfer_request', self.volume_transfer.id)]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)