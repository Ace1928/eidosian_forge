from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareNetworkShow(TestShareNetwork):

    def setUp(self):
        super(TestShareNetworkShow, self).setUp()
        self.share_network = manila_fakes.FakeShareNetwork.create_one_share_network()
        self.share_networks_mock.get.return_value = self.share_network
        self.security_services_mock = self.app.client_manager.share.security_services
        self.cmd = osc_share_networks.ShowShareNetwork(self.app, None)
        self.data = self.share_network._info.values()
        self.columns = self.share_network._info.keys()

    def test_share_network_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    @ddt.data('name', 'id')
    def test_share_network_show_by(self, attr):
        network_to_show = getattr(self.share_network, attr)
        fake_security_service = mock.Mock()
        fake_security_service.id = str(uuid.uuid4())
        fake_security_service.name = 'security-service-%s' % uuid.uuid4().hex
        self.security_services_mock.list = mock.Mock(return_value=[fake_security_service])
        arglist = [network_to_show]
        verifylist = [('share_network', network_to_show)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.find_resource', return_value=self.share_network) as find_resource:
            columns, data = self.cmd.take_action(parsed_args)
            find_resource.assert_called_once_with(self.share_networks_mock, network_to_show)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)