import testtools
from osc_lib import exceptions
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
class TestDeleteVPNaaS(test_fakes.TestNeutronClientOSCV2):

    def test_delete_with_one_resource(self):
        target = self.resource['id']

        def _mock_vpnaas(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_vpn_endpoint_group.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_ipsec_site_connection.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_ike_policy.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_ipsec_policy.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_service.side_effect = _mock_vpnaas
        arglist = [target]
        verifylist = [(self.res, [target])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target)
        self.assertIsNone(result)

    def test_delete_with_multiple_resources(self):

        def _mock_vpnaas(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_vpn_endpoint_group.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_ipsec_site_connection.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_ike_policy.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_ipsec_policy.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_service.side_effect = _mock_vpnaas
        target1 = 'target1'
        target2 = 'target2'
        arglist = [target1, target2]
        verifylist = [(self.res, [target1, target2])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.assertEqual(2, self.mocked.call_count)
        for idx, reference in enumerate([target1, target2]):
            actual = ''.join(self.mocked.call_args_list[idx][0])
            self.assertEqual(reference, actual)

    def test_delete_multiple_with_exception(self):
        target1 = 'target'
        arglist = [target1]
        verifylist = [(self.res, [target1])]
        self.networkclient.find_vpn_ipsec_site_connection.side_effect = [target1, exceptions.CommandError]
        self.networkclient.find_vpn_endpoint_group.side_effect = [target1, exceptions.CommandError]
        self.networkclient.find_vpn_ike_policy.side_effect = [target1, exceptions.CommandError]
        self.networkclient.find_vpn_service.side_effect = [target1, exceptions.CommandError]
        self.networkclient.find_vpn_ipsec_policy.side_effect = [target1, exceptions.CommandError]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        resource_name = self.res.replace('_', ' ')
        msg = '1 of 2 %s(s) failed to delete.' % resource_name
        with testtools.ExpectedException(exceptions.CommandError) as e:
            self.cmd.take_action(parsed_args)
            self.assertEqual(msg, str(e))