import copy
import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.osc.v2.networking_bgpvpn import bgpvpn
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
class TestSetBgpvpn(fakes.TestNeutronClientBgpvpn):

    def setUp(self):
        super(TestSetBgpvpn, self).setUp()
        self.networkclient.find_bgpvpn = mock.Mock(side_effect=lambda name_or_id: {'id': name_or_id})
        self.cmd = bgpvpn.SetBgpvpn(self.app, self.namespace)

    def test_set_bgpvpn(self):
        attrs = {'route_targets': ['set_rt1', 'set_rt2', 'set_rt3'], 'import_targets': ['set_irt1', 'set_irt2', 'set_irt3'], 'export_targets': ['set_ert1', 'set_ert2', 'set_ert3'], 'route_distinguishers': ['set_rd1', 'set_rd2', 'set_rd3']}
        fake_bgpvpn = fakes.create_one_bgpvpn(attrs)
        self.networkclient.get_bgpvpn = mock.Mock(return_value=fake_bgpvpn)
        self.networkclient.update_bgpvpn = mock.Mock()
        arglist = [fake_bgpvpn['id'], '--name', 'set_name', '--route-target', 'set_rt1', '--import-target', 'set_irt1', '--export-target', 'set_ert1', '--route-distinguisher', 'set_rd1']
        verifylist = [('bgpvpn', fake_bgpvpn['id']), ('name', 'set_name'), ('route_targets', ['set_rt1']), ('purge_route_target', False), ('import_targets', ['set_irt1']), ('purge_import_target', False), ('export_targets', ['set_ert1']), ('purge_export_target', False), ('route_distinguishers', ['set_rd1']), ('purge_route_distinguisher', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'set_name', 'route_targets': list(set(fake_bgpvpn['route_targets']) | set(['set_rt1'])), 'import_targets': list(set(fake_bgpvpn['import_targets']) | set(['set_irt1'])), 'export_targets': list(set(fake_bgpvpn['export_targets']) | set(['set_ert1'])), 'route_distinguishers': list(set(fake_bgpvpn['route_distinguishers']) | set(['set_rd1']))}
        self.networkclient.update_bgpvpn.assert_called_once_with(fake_bgpvpn['id'], **attrs)
        self.assertIsNone(result)

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