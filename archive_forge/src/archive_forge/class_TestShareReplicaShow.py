from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareReplicaShow(TestShareReplica):

    def setUp(self):
        super(TestShareReplicaShow, self).setUp()
        self.share_replica = manila_fakes.FakeShareReplica.create_one_replica()
        self.replicas_mock.get.return_value = self.share_replica
        self.replica_el_list = manila_fakes.FakeShareExportLocation.create_share_export_locations(count=2)
        self.replica_el_mock.list.return_value = self.replica_el_list
        self.cmd = osc_share_replicas.ShowShareReplica(self.app, None)
        self.share_replica._info['export_locations'] = cliutils.convert_dict_list_to_string(self.replica_el_list)
        self.data = tuple(self.share_replica._info.values())
        self.columns = tuple(self.share_replica._info.keys())

    def test_share_replica_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_replica_show(self):
        arglist = [self.share_replica.id]
        verifylist = [('replica', self.share_replica.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.replicas_mock.get.assert_called_with(self.share_replica.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)