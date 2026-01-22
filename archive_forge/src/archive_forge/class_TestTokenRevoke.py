from unittest import mock
from openstackclient.identity.v2_0 import token
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestTokenRevoke(TestToken):
    TOKEN = 'fob'

    def setUp(self):
        super(TestTokenRevoke, self).setUp()
        self.tokens_mock = self.app.client_manager.identity.tokens
        self.tokens_mock.reset_mock()
        self.tokens_mock.delete.return_value = True
        self.cmd = token.RevokeToken(self.app, None)

    def test_token_revoke(self):
        arglist = [self.TOKEN]
        verifylist = [('token', self.TOKEN)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.tokens_mock.delete.assert_called_with(self.TOKEN)
        self.assertIsNone(result)