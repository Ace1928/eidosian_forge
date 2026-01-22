import copy
import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallgroup
from neutronclient.osc.v2 import utils as v2_utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
class TestCreateFirewallGroup(TestFirewallGroup, common.TestCreateFWaaS):

    def setUp(self):
        super(TestCreateFirewallGroup, self).setUp()
        self.networkclient.create_firewall_group = mock.Mock(return_value=_fwg)
        self.mocked = self.networkclient.create_firewall_group
        self.cmd = firewallgroup.CreateFirewallGroup(self.app, self.namespace)

    def _update_expect_response(self, request, response):
        """Set expected request and response

        :param request
            A dictionary of request body(dict of verifylist)
        :param response
            A OrderedDict of request body
        """
        self.networkclient.create_firewall_group.return_value = response
        osc_utils.find_project.return_value.id = response['tenant_id']
        self.data = _generate_response(ordered_dict=response)
        self.expected_data = response

    def test_create_with_no_option(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.ordered_headers, tuple(sorted(headers)))

    def test_create_with_port(self):
        port_id = 'id_for_port'

        def _mock_find(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_port.side_effect = _mock_find
        arglist = ['--port', port_id]
        verifylist = [('port', [port_id])]
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.check_results(headers, data, request)

    def test_create_with_ingress_policy(self):
        ingress_policy = 'my-ingress-policy'

        def _mock_port_fwg(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_firewall_policy.side_effect = _mock_port_fwg
        arglist = ['--ingress-firewall-policy', ingress_policy]
        verifylist = [('ingress_firewall_policy', ingress_policy)]
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.networkclient.find_firewall_policy.assert_called_once_with(ingress_policy)
        self.check_results(headers, data, request)

    def test_create_with_egress_policy(self):
        egress_policy = 'my-egress-policy'

        def _mock_find(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_firewall_group.side_effect = _mock_find
        self.networkclient.find_firewall_policy.side_effect = _mock_find
        arglist = ['--egress-firewall-policy', egress_policy]
        verifylist = [('egress_firewall_policy', egress_policy)]
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.networkclient.find_firewall_policy.assert_called_once_with(egress_policy)
        self.check_results(headers, data, request)

    def test_create_with_all_params(self):
        name = 'my-name'
        description = 'my-desc'
        ingress_policy = 'my-ingress-policy'
        egress_policy = 'my-egress-policy'

        def _mock_find(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_firewall_policy.side_effect = _mock_find
        port = 'port'
        self.networkclient.find_port.side_effect = _mock_find
        tenant_id = 'my-tenant'
        arglist = ['--name', name, '--description', description, '--ingress-firewall-policy', ingress_policy, '--egress-firewall-policy', egress_policy, '--port', port, '--project', tenant_id, '--share', '--disable']
        verifylist = [('name', name), ('description', description), ('ingress_firewall_policy', ingress_policy), ('egress_firewall_policy', egress_policy), ('port', [port]), ('share', True), ('project', tenant_id), ('disable', True)]
        request, response = _generate_req_and_res(verifylist)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.check_results(headers, data, request)

    def test_create_with_shared_and_no_share(self):
        arglist = ['--share', '--no-share']
        verifylist = [('share', True), ('no_share', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_ports_and_no(self):
        port = 'my-port'
        arglist = ['--port', port, '--no-port']
        verifylist = [('port', [port]), ('no_port', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_ingress_policy_and_no(self):
        policy = 'my-policy'
        arglist = ['--ingress-firewall-policy', policy, '--no-ingress-firewall-policy']
        verifylist = [('ingress_firewall_policy', policy), ('no_ingress_firewall_policy', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_egress_policy_and_no(self):
        policy = 'my-policy'
        arglist = ['--egress-firewall-policy', policy, '--no-egress-firewall-policy']
        verifylist = [('egress_firewall_policy', policy), ('no_egress_firewall_policy', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)