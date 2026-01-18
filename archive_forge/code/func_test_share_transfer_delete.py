from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_transfers as osc_share_transfers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_transfer_delete(self):
    arglist = [self.transfer.id]
    verifylist = [('transfer', [self.transfer.id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.transfers_mock.delete.assert_called_with(self.transfer.id)
    self.assertIsNone(result)