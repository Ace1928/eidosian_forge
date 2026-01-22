from unittest import mock
from osc_lib.cli import format_columns
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsec_site_connection
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestCreateIPsecSiteConn(TestIPsecSiteConn, common.TestCreateVPNaaS):

    def setUp(self):
        super(TestCreateIPsecSiteConn, self).setUp()
        self.networkclient.create_vpn_ipsec_site_connection = mock.Mock(return_value=_ipsec_site_conn)
        self.mocked = self.networkclient.create_vpn_ipsec_site_connection
        self.cmd = ipsec_site_connection.CreateIPsecSiteConnection(self.app, self.namespace)

    def _update_expect_response(self, request, response):
        """Set expected request and response

        :param request
            A dictionary of request body(dict of verifylist)
        :param response
            A OrderedDict of request body
        """
        self.networkclient.create_vpn_ipsec_site_connection.return_value = response
        osc_utils.find_project.return_value.id = response['project_id']
        self.data = _generate_data(ordered_dict=response)
        self.ordered_data = (response['auth_mode'], response['dpd'], response['description'], response['id'], response['ikepolicy_id'], response['ipsecpolicy_id'], response['initiator'], response['local_ep_group_id'], response['local_id'], response['mtu'], response['name'], response['peer_address'], format_columns.ListColumn(response['peer_cidrs']), response['peer_ep_group_id'], response['peer_id'], response['psk'], response['project_id'], response['route_mode'], response['admin_state_up'], response['status'], response['vpnservice_id'])

    def _set_all_params(self, args={}):
        tenant_id = args.get('tenant_id') or 'my-tenant'
        name = args.get('name') or 'connection1'
        peer_address = args.get('peer_address') or '192.168.2.10'
        peer_id = args.get('peer_id') or '192.168.2.10'
        psk = args.get('psk') or 'abcd'
        mtu = args.get('mtu') or '1500'
        initiator = args.get('initiator') or 'bi-directional'
        vpnservice_id = args.get('vpnservice') or 'vpnservice_id'
        ikepolicy_id = args.get('ikepolicy') or 'ikepolicy_id'
        ipsecpolicy_id = args.get('ipsecpolicy') or 'ipsecpolicy_id'
        local_ep_group = args.get('local_ep_group_id') or 'local-epg'
        peer_ep_group = args.get('peer_ep_group_id') or 'peer-epg'
        description = args.get('description') or 'my-vpn-connection'
        arglist = ['--project', tenant_id, '--peer-address', peer_address, '--peer-id', peer_id, '--psk', psk, '--initiator', initiator, '--vpnservice', vpnservice_id, '--ikepolicy', ikepolicy_id, '--ipsecpolicy', ipsecpolicy_id, '--mtu', mtu, '--description', description, '--local-endpoint-group', local_ep_group, '--peer-endpoint-group', peer_ep_group, name]
        verifylist = [('project', tenant_id), ('peer_address', peer_address), ('peer_id', peer_id), ('psk', psk), ('initiator', initiator), ('vpnservice', vpnservice_id), ('ikepolicy', ikepolicy_id), ('ipsecpolicy', ipsecpolicy_id), ('mtu', mtu), ('description', description), ('local_endpoint_group', local_ep_group), ('peer_endpoint_group', peer_ep_group), ('name', name)]
        return (arglist, verifylist)

    def _test_create_with_all_params(self, args={}):
        arglist, verifylist = self._set_all_params(args)
        request, response = _generate_req_and_res(verifylist)

        def _mock_endpoint_group(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_vpn_endpoint_group.side_effect = mock.Mock(side_effect=_mock_endpoint_group)
        self.networkclient.find_vpn_service.side_effect = mock.Mock(side_effect=_mock_endpoint_group)
        self.networkclient.find_vpn_ike_policy.side_effect = mock.Mock(side_effect=_mock_endpoint_group)
        self.networkclient.find_vpn_ipsec_policy.side_effect = mock.Mock(side_effect=_mock_endpoint_group)
        self._update_expect_response(request, response)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.check_results(headers, data, request)

    def test_create_with_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_all_params(self):
        self._test_create_with_all_params()