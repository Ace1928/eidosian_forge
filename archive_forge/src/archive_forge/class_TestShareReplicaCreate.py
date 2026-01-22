from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareReplicaCreate(TestShareReplica):

    def setUp(self):
        super(TestShareReplicaCreate, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share()
        self.shares_mock.get.return_value = self.share
        self.share_replica = manila_fakes.FakeShareReplica.create_one_replica(attrs={'availability_zone': 'manila-zone-1', 'status': 'available'})
        self.replicas_mock.create.return_value = self.share_replica
        self.replicas_mock.get.return_value = self.share_replica
        self.cmd = osc_share_replicas.CreateShareReplica(self.app, None)
        self.data = tuple(self.share_replica._info.values())
        self.columns = tuple(self.share_replica._info.keys())

    def test_share_replica_create_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_replica_create(self):
        arglist = [self.share.id]
        verifylist = [('share', self.share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.replicas_mock.create.assert_called_with(share=self.share, availability_zone=None)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_replica_create_az(self):
        arglist = [self.share.id, '--availability-zone', self.share.availability_zone]
        verifylist = [('share', self.share.id), ('availability_zone', self.share.availability_zone)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.replicas_mock.create.assert_called_with(share=self.share, availability_zone=self.share.availability_zone)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_replica_create_scheduler_hint_valid(self):
        arglist = [self.share.id, '--availability-zone', self.share.availability_zone, '--scheduler-hint', 'only_host=host1@backend1#pool1']
        verifylist = [('share', self.share.id), ('availability_zone', self.share.availability_zone), ('scheduler_hint', {'only_host': 'host1@backend1#pool1'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.replicas_mock.create.assert_called_with(share=self.share, availability_zone=self.share.availability_zone, scheduler_hints={'only_host': 'host1@backend1#pool1'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_replica_create_scheduler_hint_invalid_hint(self):
        arglist = [self.share.id, '--availability-zone', self.share.availability_zone, '--scheduler-hint', 'fake_hint=host1@backend1#pool1']
        verifylist = [('share', self.share.id), ('availability_zone', self.share.availability_zone), ('scheduler_hint', {'fake_hint': 'host1@backend1#pool1'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_replica_create_scheduler_hint_invalid_version(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.66')
        arglist = [self.share.id, '--availability-zone', self.share.availability_zone, '--scheduler-hint', 'only_host=host1@backend1#pool1']
        verifylist = [('share', self.share.id), ('availability_zone', self.share.availability_zone), ('scheduler_hint', {'only_host': 'host1@backend1#pool1'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_replica_create_share_network(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.72')
        arglist = [self.share.id, '--availability-zone', self.share.availability_zone, '--share-network', self.share.share_network_id]
        verifylist = [('share', self.share.id), ('availability_zone', self.share.availability_zone), ('share_network', self.share.share_network_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        if self.share.share_network_id:
            self.replicas_mock.create.assert_called_with(share=self.share, availability_zone=self.share.availability_zone, share_network=self.share.share_network_id)
        else:
            self.replicas_mock.create.assert_called_with(share=self.share, availability_zone=self.share.availability_zone)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_replica_create_wait(self):
        arglist = [self.share.id, '--wait']
        verifylist = [('share', self.share.id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.replicas_mock.create.assert_called_with(share=self.share, availability_zone=None)
        self.replicas_mock.get.assert_called_with(self.share_replica.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    @mock.patch('manilaclient.osc.v2.share_replicas.LOG')
    def test_share_replica_create_wait_exception(self, mock_logger):
        arglist = [self.share.id, '--wait']
        verifylist = [('share', self.share.id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
            columns, data = self.cmd.take_action(parsed_args)
            self.replicas_mock.create.assert_called_with(share=self.share, availability_zone=None)
            mock_logger.error.assert_called_with('ERROR: Share replica is in error state.')
            self.replicas_mock.get.assert_called_with(self.share_replica.id)
            self.assertCountEqual(self.columns, columns)
            self.assertCountEqual(self.data, data)