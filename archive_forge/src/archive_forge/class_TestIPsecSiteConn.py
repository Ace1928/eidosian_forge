from unittest import mock
from osc_lib.cli import format_columns
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsec_site_connection
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestIPsecSiteConn(test_fakes.TestNeutronClientOSCV2):

    def check_results(self, headers, data, exp_req, is_list=False):
        if is_list:
            req_body = {self.res_plural: list(exp_req)}
        else:
            req_body = exp_req
        self.mocked.assert_called_once_with(**req_body)
        self.assertEqual(self.ordered_headers, tuple(sorted(headers)))
        self.assertItemEqual(self.ordered_data, data)

    def setUp(self):
        super(TestIPsecSiteConn, self).setUp()

        def _mock_ipsec_site_conn(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_vpn_ipsec_site_connection.side_effect = mock.Mock(side_effect=_mock_ipsec_site_conn)
        osc_utils.find_project = mock.Mock()
        osc_utils.find_project.id = _ipsec_site_conn['project_id']
        self.res = 'ipsec_site_connection'
        self.res_plural = 'ipsec_site_connections'
        self.resource = _ipsec_site_conn
        self.headers = ('ID', 'Name', 'Peer Address', 'Authentication Algorithm', 'Status', 'Project', 'Peer CIDRs', 'VPN Service', 'IPSec Policy', 'IKE Policy', 'MTU', 'Initiator', 'State', 'Description', 'Pre-shared Key', 'Route Mode', 'Local ID', 'Peer ID', 'Local Endpoint Group ID', 'Peer Endpoint Group ID', 'DPD')
        self.data = _generate_data()
        self.ordered_headers = ('Authentication Algorithm', 'DPD', 'Description', 'ID', 'IKE Policy', 'IPSec Policy', 'Initiator', 'Local Endpoint Group ID', 'Local ID', 'MTU', 'Name', 'Peer Address', 'Peer CIDRs', 'Peer Endpoint Group ID', 'Peer ID', 'Pre-shared Key', 'Project', 'Route Mode', 'State', 'Status', 'VPN Service')
        self.ordered_data = (_ipsec_site_conn['auth_mode'], _ipsec_site_conn['dpd'], _ipsec_site_conn['description'], _ipsec_site_conn['id'], _ipsec_site_conn['ikepolicy_id'], _ipsec_site_conn['ipsecpolicy_id'], _ipsec_site_conn['initiator'], _ipsec_site_conn['local_ep_group_id'], _ipsec_site_conn['local_id'], _ipsec_site_conn['mtu'], _ipsec_site_conn['name'], _ipsec_site_conn['peer_address'], format_columns.ListColumn(_ipsec_site_conn['peer_cidrs']), _ipsec_site_conn['peer_ep_group_id'], _ipsec_site_conn['peer_id'], _ipsec_site_conn['psk'], _ipsec_site_conn['project_id'], _ipsec_site_conn['route_mode'], _ipsec_site_conn['admin_state_up'], _ipsec_site_conn['status'], _ipsec_site_conn['vpnservice_id'])