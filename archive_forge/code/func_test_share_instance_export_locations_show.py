from osc_lib import utils as osc_lib_utils
from manilaclient.osc.v2 \
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_instance_export_locations_show(self):
    arglist = [self.instance.id, self.share_instance_export_locations.id]
    verifylist = [('instance', self.instance.id), ('export_location', self.share_instance_export_locations.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.instances_mock.get.assert_called_with(self.instance.id)
    self.instance_export_locations_mock.get.assert_called_with(self.instance.id, self.share_instance_export_locations.id)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)