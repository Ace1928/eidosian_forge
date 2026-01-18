import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc import utils
from manilaclient.osc.v2 import services as osc_services
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_service_set_disable(self):
    arglist = [self.share_service.host, self.share_service.binary, '--disable']
    verifylist = [('host', self.share_service.host), ('binary', self.share_service.binary), ('disable', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.services_mock.disable.assert_called_with(self.share_service.host, self.share_service.binary)
    self.assertIsNone(result)