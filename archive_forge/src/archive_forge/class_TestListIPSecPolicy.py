from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsecpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestListIPSecPolicy(TestIPSecPolicy):

    def setUp(self):
        super(TestListIPSecPolicy, self).setUp()
        self.cmd = ipsecpolicy.ListIPsecPolicy(self.app, self.namespace)
        self.short_header = ('ID', 'Name', 'Authentication Algorithm', 'Encapsulation Mode', 'Transform Protocol', 'Encryption Algorithm')
        self.short_data = (_ipsecpolicy['id'], _ipsecpolicy['name'], _ipsecpolicy['auth_algorithm'], _ipsecpolicy['encapsulation_mode'], _ipsecpolicy['transform_protocol'], _ipsecpolicy['encryption_algorithm'])
        self.networkclient.vpn_ipsec_policies = mock.Mock(return_value=[_ipsecpolicy])
        self.mocked = self.networkclient.vpn_ipsec_policies

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