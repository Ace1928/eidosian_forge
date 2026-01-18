import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_security_service_list_ad_site_api_version_exception(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.75')
    arglist = ['--default-ad-site', self.services_list[0].default_ad_site]
    verifylist = [('default_ad_site', self.services_list[0].default_ad_site)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)