import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc import utils
from manilaclient.osc.v2 import services as osc_services
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_service_set_disable_with_reason(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.83')
    reason = 'earthquake'
    arglist = ['--disable', '--disable-reason', reason, self.share_service.host, self.share_service.binary]
    verifylist = [('host', self.share_service.host), ('binary', self.share_service.binary), ('disable', True), ('disable_reason', reason)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.services_mock.disable.assert_called_with(self.share_service.host, self.share_service.binary, disable_reason=reason)
    self.assertIsNone(result)