from openstackclient.identity.v3 import domain
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestDomainDelete(TestDomain):
    domain = identity_fakes.FakeDomain.create_one_domain()

    def setUp(self):
        super(TestDomainDelete, self).setUp()
        self.domains_mock.get.return_value = self.domain
        self.domains_mock.delete.return_value = None
        self.cmd = domain.DeleteDomain(self.app, None)

    def test_domain_delete(self):
        arglist = [self.domain.id]
        verifylist = [('domain', [self.domain.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.domains_mock.delete.assert_called_with(self.domain.id)
        self.assertIsNone(result)