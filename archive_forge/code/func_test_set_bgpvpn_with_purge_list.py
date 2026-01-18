import copy
import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.osc.v2.networking_bgpvpn import bgpvpn
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_set_bgpvpn_with_purge_list(self):
    fake_bgpvpn = fakes.create_one_bgpvpn()
    self.networkclient.get_bgpvpn = mock.Mock(return_value=fake_bgpvpn)
    self.neutronclient.update_bgpvpn = mock.Mock()
    arglist = [fake_bgpvpn['id'], '--route-target', 'set_rt1', '--no-route-target', '--import-target', 'set_irt1', '--no-import-target', '--export-target', 'set_ert1', '--no-export-target', '--route-distinguisher', 'set_rd1', '--no-route-distinguisher']
    verifylist = [('bgpvpn', fake_bgpvpn['id']), ('route_targets', ['set_rt1']), ('purge_route_target', True), ('import_targets', ['set_irt1']), ('purge_import_target', True), ('export_targets', ['set_ert1']), ('purge_export_target', True), ('route_distinguishers', ['set_rd1']), ('purge_route_distinguisher', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'route_targets': [], 'import_targets': [], 'export_targets': [], 'route_distinguishers': []}
    self.networkclient.update_bgpvpn.assert_called_once_with(fake_bgpvpn['id'], **attrs)
    self.assertIsNone(result)