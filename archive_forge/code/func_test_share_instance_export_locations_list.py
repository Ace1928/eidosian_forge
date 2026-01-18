from osc_lib import utils as osc_lib_utils
from manilaclient.osc.v2 \
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_instance_export_locations_list(self):
    arglist = [self.instance.id]
    verifylist = [('instance', self.instance.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.instances_mock.get.assert_called_with(self.instance.id)
    self.instance_export_locations_mock.list.assert_called_with(self.instance, search_opts=None)
    self.assertCountEqual(self.column_headers, columns)
    self.assertCountEqual(self.data, data)