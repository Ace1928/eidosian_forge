from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data({'status': 'error', 'current_security_service': str(uuid.uuid4()), 'check_only': True, 'restart_check': True}, {'status': None, 'current_security_service': str(uuid.uuid4()), 'check_only': True, 'restart_check': None}, {'status': None, 'current_security_service': str(uuid.uuid4()), 'check_only': True, 'restart_check': True})
@ddt.unpack
def test_set_share_network_api_version_exception(self, status, current_security_service, check_only, restart_check):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.62')
    arglist = [self.share_network.id]
    verifylist = [('share_network', self.share_network.id)]
    if status:
        arglist.extend(['--status', status])
        verifylist.append(('status', status))
    if current_security_service:
        arglist.extend(['--current-security-service', current_security_service])
        verifylist.append(('current_security_service', current_security_service))
    if check_only and restart_check:
        arglist.extend(['--check-only', '--restart-check'])
        verifylist.extend([('check_only', True), ('restart_check', True)])
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)