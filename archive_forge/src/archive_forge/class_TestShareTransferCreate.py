from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_transfers as osc_share_transfers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareTransferCreate(TestShareTransfer):

    def setUp(self):
        super(TestShareTransferCreate, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share()
        self.shares_mock.create.return_value = self.share
        self.shares_mock.get.return_value = self.share
        self.share_transfer = manila_fakes.FakeShareTransfer.create_one_transfer()
        self.transfers_mock.get.return_value = self.share_transfer
        self.transfers_mock.create.return_value = self.share_transfer
        self.cmd = osc_share_transfers.CreateShareTransfer(self.app, None)
        self.data = tuple(self.share_transfer._info.values())
        self.columns = tuple(self.share_transfer._info.keys())

    def test_share_transfer_create_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_transfer_create_required_args(self):
        arglist = [self.share.id]
        verifylist = [('share', self.share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.transfers_mock.create.assert_called_with(self.share.id, name=None)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)