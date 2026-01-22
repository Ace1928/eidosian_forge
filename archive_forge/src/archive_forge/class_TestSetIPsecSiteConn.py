from unittest import mock
from osc_lib.cli import format_columns
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsec_site_connection
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestSetIPsecSiteConn(TestIPsecSiteConn, common.TestSetVPNaaS):

    def setUp(self):
        super(TestSetIPsecSiteConn, self).setUp()
        self.networkclient.update_vpn_ipsec_site_connection = mock.Mock(return_value=_ipsec_site_conn)
        self.mocked = self.networkclient.update_vpn_ipsec_site_connection
        self.cmd = ipsec_site_connection.SetIPsecSiteConnection(self.app, self.namespace)

    def test_set_ipsec_site_conn_with_peer_id(self):
        target = self.resource['id']
        peer_id = '192.168.3.10'
        arglist = [target, '--peer-id', peer_id]
        verifylist = [(self.res, target), ('peer_id', peer_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'peer_id': peer_id})
        self.assertIsNone(result)