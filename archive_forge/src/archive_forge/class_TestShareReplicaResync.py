from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareReplicaResync(TestShareReplica):

    def setUp(self):
        super(TestShareReplicaResync, self).setUp()
        self.share_replica = manila_fakes.FakeShareReplica.create_one_replica()
        self.replicas_mock.get.return_value = self.share_replica
        self.cmd = osc_share_replicas.ResyncShareReplica(self.app, None)

    def test_share_replica_resync(self):
        arglist = [self.share_replica.id]
        verifylist = [('replica', self.share_replica.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.replicas_mock.resync.assert_called_with(self.share_replica)
        self.assertIsNone(result)

    def test_share_replica_resync_exception(self):
        arglist = [self.share_replica.id]
        verifylist = [('replica', self.share_replica.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.replicas_mock.resync.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)