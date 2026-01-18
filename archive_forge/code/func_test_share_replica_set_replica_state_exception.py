from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_replica_set_replica_state_exception(self):
    new_replica_state = 'in_sync'
    arglist = [self.share_replica.id, '--replica-state', new_replica_state]
    verifylist = [('replica', self.share_replica.id), ('replica_state', new_replica_state)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.replicas_mock.reset_replica_state.side_effect = Exception()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)