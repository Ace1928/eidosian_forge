from osc_lib import utils as osc_lib_utils
from manilaclient.osc.v2 \
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareInstanceExportLocationList(TestShareInstanceExportLocation):
    column_headers = ['ID', 'Path', 'Is Admin Only', 'Preferred']

    def setUp(self):
        super(TestShareInstanceExportLocationList, self).setUp()
        self.instance = manila_fakes.FakeShareInstance.create_one_share_instance()
        self.instances_mock.get.return_value = self.instance
        self.instance_export_locations = manila_fakes.FakeShareExportLocation.create_share_export_locations()
        self.instance_export_locations_mock.list.return_value = self.instance_export_locations
        self.data = (osc_lib_utils.get_dict_properties(i._info, self.column_headers) for i in self.instance_export_locations)
        self.cmd = osc_share_instance_export_locations.ShareInstanceListExportLocation(self.app, None)

    def test_share_instance_export_locations_list_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_instance_export_locations_list(self):
        arglist = [self.instance.id]
        verifylist = [('instance', self.instance.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.instances_mock.get.assert_called_with(self.instance.id)
        self.instance_export_locations_mock.list.assert_called_with(self.instance, search_opts=None)
        self.assertCountEqual(self.column_headers, columns)
        self.assertCountEqual(self.data, data)