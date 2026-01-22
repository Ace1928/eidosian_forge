from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsecpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestSetIPSecPolicy(TestIPSecPolicy, common.TestSetVPNaaS):

    def setUp(self):
        super(TestSetIPSecPolicy, self).setUp()
        self.networkclient.update_vpn_ipsec_policy = mock.Mock(return_value=_ipsecpolicy)
        self.mocked = self.networkclient.update_vpn_ipsec_policy
        self.cmd = ipsecpolicy.SetIPsecPolicy(self.app, self.namespace)

    def test_set_auth_algorithm_with_sha256(self):
        target = self.resource['id']
        auth_algorithm = 'sha256'
        arglist = [target, '--auth-algorithm', auth_algorithm]
        verifylist = [(self.res, target), ('auth_algorithm', auth_algorithm)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'auth_algorithm': 'sha256'})
        self.assertIsNone(result)