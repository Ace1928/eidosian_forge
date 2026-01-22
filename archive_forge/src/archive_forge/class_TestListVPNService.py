from unittest import mock
import uuid
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import vpnservice
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestListVPNService(TestVPNService):

    def setUp(self):
        super(TestListVPNService, self).setUp()
        self.cmd = vpnservice.ListVPNService(self.app, self.namespace)
        self.short_header = ('ID', 'Name', 'Router', 'Subnet', 'Flavor', 'State', 'Status')
        self.short_data = (_vpnservice['id'], _vpnservice['name'], _vpnservice['router_id'], _vpnservice['subnet_id'], _vpnservice['flavor_id'], _vpnservice['admin_state_up'], _vpnservice['status'])
        self.networkclient.vpn_services = mock.Mock(return_value=[_vpnservice])
        self.mocked = self.networkclient.vpn_services

    def test_list_with_long_option(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.headers), headers)

    def test_list_with_no_option(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.short_header), headers)
        self.assertEqual([self.short_data], list(data))