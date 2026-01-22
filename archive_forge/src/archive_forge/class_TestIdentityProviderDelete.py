import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
class TestIdentityProviderDelete(TestIdentityProvider):

    def setUp(self):
        super(TestIdentityProviderDelete, self).setUp()
        self.identity_providers_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.IDENTITY_PROVIDER), loaded=True)
        self.identity_providers_mock.delete.return_value = None
        self.cmd = identity_provider.DeleteIdentityProvider(self.app, None)

    def test_delete_identity_provider(self):
        arglist = [identity_fakes.idp_id]
        verifylist = [('identity_provider', [identity_fakes.idp_id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.identity_providers_mock.delete.assert_called_with(identity_fakes.idp_id)
        self.assertIsNone(result)