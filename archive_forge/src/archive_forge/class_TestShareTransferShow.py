from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_transfers as osc_share_transfers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareTransferShow(TestShareTransfer):

    def setUp(self):
        super(TestShareTransferShow, self).setUp()
        self.transfer = manila_fakes.FakeShareTransfer.create_one_transfer()
        self.transfers_mock.get.return_value = self.transfer
        self.cmd = osc_share_transfers.ShowShareTransfer(self.app, None)
        self.data = self.transfer._info.values()
        self.transfer._info.pop('auth_key')
        self.columns = self.transfer._info.keys()

    def test_share_transfer_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_transfer_show(self):
        arglist = [self.transfer.id]
        verifylist = [('transfer', self.transfer.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.transfers_mock.get.assert_called_with(self.transfer.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)