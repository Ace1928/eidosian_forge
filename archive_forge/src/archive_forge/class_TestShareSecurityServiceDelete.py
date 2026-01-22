import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSecurityServiceDelete(TestShareSecurityService):

    def setUp(self):
        super(TestShareSecurityServiceDelete, self).setUp()
        self.security_service = manila_fakes.FakeShareSecurityService.create_fake_security_service()
        self.security_services_mock.get.return_value = self.security_service
        self.security_services = manila_fakes.FakeShareSecurityService.create_fake_security_services()
        self.cmd = osc_security_services.DeleteShareSecurityService(self.app, None)

    def test_share_security_service_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_security_service_delete(self):
        arglist = [self.security_services[0].id, self.security_services[1].id]
        verifylist = [('security_service', [self.security_services[0].id, self.security_services[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.security_services_mock.delete.call_count, len(self.security_services))
        self.assertIsNone(result)

    def test_share_security_service_delete_exception(self):
        arglist = [self.security_services[0].id]
        verifylist = [('security_service', [self.security_services[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.security_services_mock.delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)