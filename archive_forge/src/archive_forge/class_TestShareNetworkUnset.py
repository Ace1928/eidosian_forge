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
class TestShareNetworkUnset(TestShareNetwork):

    def setUp(self):
        super(TestShareNetworkUnset, self).setUp()
        self.share_network = manila_fakes.FakeShareNetwork.create_one_share_network()
        self.share_networks_mock.get.return_value = self.share_network
        self.cmd = osc_share_networks.UnsetShareNetwork(self.app, None)

    def test_unset_share_network_name(self):
        arglist = [self.share_network.id, '--name']
        verifylist = [('share_network', self.share_network.id), ('name', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.find_resource', return_value=self.share_network):
            result = self.cmd.take_action(parsed_args)
        self.share_networks_mock.update.assert_called_once_with(self.share_network, name='')
        self.assertIsNone(result)

    def test_unset_share_network_description(self):
        arglist = [self.share_network.id, '--description']
        verifylist = [('share_network', self.share_network.id), ('description', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.share_networks_mock.update.assert_called_once_with(self.share_network, description='')
        self.assertIsNone(result)

    @ddt.data('name', 'security_service')
    def test_unset_share_network_exception_while_updating(self, attr):
        arglist = [self.share_network.id, '--name', '--security-service', 'fake-security-service']
        verifylist = [('share_network', self.share_network.id), ('name', True), ('security_service', 'fake-security-service')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        if attr == 'name':
            self.share_networks_mock.update.side_effect = Exception()
        else:
            self.share_networks_mock.remove_security_service.side_effect = Exception()
        with mock.patch('osc_lib.utils.find_resource', side_effect=[self.share_network, 'fake-security-service']):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.share_networks_mock.update.assert_called_once_with(self.share_network, name='')
        if attr == 'security_service':
            self.share_networks_mock.remove_security_service.assert_called_once_with(self.share_network, 'fake-security-service')

    def test_unset_share_network_security_service(self):
        arglist = [self.share_network.id, '--security-service', 'fake-security-service']
        verifylist = [('share_network', self.share_network.id), ('security_service', 'fake-security-service')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.find_resource', side_effect=[self.share_network, 'fake-security-service']):
            result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.share_networks_mock.remove_security_service.assert_called_once_with(self.share_network, 'fake-security-service')