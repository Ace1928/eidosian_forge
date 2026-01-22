from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ikepolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestSetIKEPolicy(TestIKEPolicy, common.TestSetVPNaaS):

    def setUp(self):
        super(TestSetIKEPolicy, self).setUp()
        self.networkclient.update_vpn_ike_policy = mock.Mock(return_value=_ikepolicy)
        self.mocked = self.networkclient.update_vpn_ike_policy
        self.cmd = ikepolicy.SetIKEPolicy(self.app, self.namespace)

    def test_set_auth_algorithm_with_sha256(self):
        target = self.resource['id']
        auth_algorithm = 'sha256'
        arglist = [target, '--auth-algorithm', auth_algorithm]
        verifylist = [(self.res, target), ('auth_algorithm', auth_algorithm)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'auth_algorithm': 'sha256'})
        self.assertIsNone(result)

    def test_set_phase1_negotiation_mode_with_aggressive(self):
        target = self.resource['id']
        phase1_negotiation_mode = 'aggressive'
        arglist = [target, '--phase1-negotiation-mode', phase1_negotiation_mode]
        verifylist = [(self.res, target), ('phase1_negotiation_mode', phase1_negotiation_mode)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'phase1_negotiation_mode': 'aggressive'})
        self.assertIsNone(result)