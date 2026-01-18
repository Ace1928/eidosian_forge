from osc_lib import utils as oscutils
from manilaclient.osc.v2 import availability_zones as osc_availability_zones
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_list_availability_zone(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.COLUMNS, columns)
    self.assertCountEqual(list(self.values), list(data))