import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareSecurityServiceSet(TestShareSecurityService):

    def setUp(self):
        super(TestShareSecurityServiceSet, self).setUp()
        self.security_service = manila_fakes.FakeShareSecurityService.create_fake_security_service(methods={'update': None})
        self.security_services_mock.get.return_value = self.security_service
        self.cmd = osc_security_services.SetShareSecurityService(self.app, None)

    def test_share_security_service_set_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_security_service_set(self):
        arglist = [self.security_service.id, '--dns-ip', self.security_service.dns_ip, '--ou', self.security_service.ou, '--server', self.security_service.server, '--domain', self.security_service.domain, '--user', self.security_service.user, '--password', self.security_service.password, '--name', self.security_service.name, '--description', self.security_service.description, '--default-ad-site', self.security_service.default_ad_site]
        verifylist = [('security_service', self.security_service.id), ('dns_ip', self.security_service.dns_ip), ('ou', self.security_service.ou), ('server', self.security_service.server), ('domain', self.security_service.domain), ('user', self.security_service.user), ('password', self.security_service.password), ('name', self.security_service.name), ('description', self.security_service.description), ('default_ad_site', self.security_service.default_ad_site)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.security_service.update.assert_called_with(dns_ip=self.security_service.dns_ip, server=self.security_service.server, domain=self.security_service.domain, user=self.security_service.user, password=self.security_service.password, name=self.security_service.name, description=self.security_service.description, ou=self.security_service.ou, default_ad_site=self.security_service.default_ad_site)
        self.assertIsNone(result)

    def test_share_security_service_set_exception(self):
        arglist = [self.security_service.id, '--name', self.security_service.name]
        verifylist = [('security_service', self.security_service.id), ('name', self.security_service.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.security_service.update.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    @ddt.data('2.43', '2.75')
    def test_share_security_service_set_api_version_exception(self, version):
        self.app.client_manager.share.api_version = api_versions.APIVersion(version)
        arglist = [self.security_service.id]
        verifylist = [('security_service', self.security_service.id)]
        if api_versions.APIVersion(version) <= api_versions.APIVersion('2.43'):
            arglist.extend(['--ou', self.security_service.ou])
            verifylist.append(('ou', self.security_service.ou))
        if api_versions.APIVersion(version) <= api_versions.APIVersion('2.75'):
            arglist.extend(['--default-ad-site', self.security_service.default_ad_site])
            verifylist.append(('default_ad_site', self.security_service.default_ad_site))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)