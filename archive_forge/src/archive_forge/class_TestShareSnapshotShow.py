from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSnapshotShow(TestShareSnapshot):

    def setUp(self):
        super(TestShareSnapshotShow, self).setUp()
        self.export_location = manila_fakes.FakeShareExportLocation.create_one_export_location()
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot(attrs={'export_locations': self.export_location})
        self.snapshots_mock.get.return_value = self.share_snapshot
        self.cmd = osc_share_snapshots.ShowShareSnapshot(self.app, None)
        self.data = self.share_snapshot._info.values()
        self.columns = self.share_snapshot._info.keys()
        self.export_locations_mock.list.return_value = [self.export_location]

    def test_share_snapshot_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_snapshot_show(self):
        arglist = [self.share_snapshot.id]
        verifylist = [('snapshot', self.share_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cliutils.convert_dict_list_to_string = mock.Mock()
        cliutils.convert_dict_list_to_string.return_value = self.export_location
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.get.assert_called_with(self.share_snapshot.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)