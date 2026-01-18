from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data({'check_only': False, 'restart_check': False}, {'check_only': True, 'restart_check': True}, {'check_only': True, 'restart_check': False})
@ddt.unpack
def test_set_share_network_add_new_security_service_check_reset(self, check_only, restart_check):
    self.share_networks_mock.add_security_service_check = mock.Mock(return_value=(200, {'compatible': True}))
    arglist = [self.share_network.id, '--new-security-service', 'new-security-service-name']
    verifylist = [('share_network', self.share_network.id), ('new_security_service', 'new-security-service-name')]
    if check_only:
        arglist.append('--check-only')
        verifylist.append(('check_only', True))
    if restart_check:
        arglist.append('--restart-check')
        verifylist.append(('restart_check', True))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.find_resource', side_effect=[self.share_network, 'new-security-service']):
        result = self.cmd.take_action(parsed_args)
    if check_only:
        self.share_networks_mock.add_security_service_check.assert_called_once_with(self.share_network, 'new-security-service', reset_operation=restart_check)
        self.share_networks_mock.add_security_service.assert_not_called()
    else:
        self.share_networks_mock.add_security_service_check.assert_not_called()
        self.share_networks_mock.add_security_service.assert_called_once_with(self.share_network, 'new-security-service')
    self.assertIsNone(result)