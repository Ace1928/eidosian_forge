from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_transfers as osc_share_transfers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_transfer_create_required_args(self):
    arglist = [self.share.id]
    verifylist = [('share', self.share.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.transfers_mock.create.assert_called_with(self.share.id, name=None)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)