from osc_lib import utils as osc_lib_utils
from manilaclient.osc.v2 import (share_snapshot_instance_export_locations as
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSnapshotInstanceExportLocationShow(TestShareSnapshotInstanceExportLocation):

    def setUp(self):
        super(TestShareSnapshotInstanceExportLocationShow, self).setUp()
        self.share_snapshot_instance = manila_fakes.FakeShareSnapshotIntances.create_one_snapshot_instance()
        self.share_snapshot_instances_export_location = manila_fakes.FakeShareSnapshotInstancesExportLocations.create_one_snapshot_instance()
        self.share_snapshot_instances_mock.get.return_value = self.share_snapshot_instance
        self.share_snapshot_instances_el_mock.get.return_value = self.share_snapshot_instances_export_location
        self.cmd = osc_snapshot_instance_locations.ShareSnapshotInstanceExportLocationShow(self.app, None)
        self.data = self.share_snapshot_instances_export_location._info.values()
        self.columns = self.share_snapshot_instances_export_location._info.keys()

    def test_share_snapshot_instance_export_location_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_snapshot_instance_export_location_show(self):
        arglist = [self.share_snapshot_instance.id, self.share_snapshot_instances_export_location.id]
        verifylist = [('snapshot_instance', self.share_snapshot_instance.id), ('export_location', self.share_snapshot_instances_export_location.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.share_snapshot_instances_mock.get.assert_called_with(self.share_snapshot_instance.id)
        self.share_snapshot_instances_el_mock.get.assert_called_with(self.share_snapshot_instances_export_location.id, snapshot_instance=self.share_snapshot_instance)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)