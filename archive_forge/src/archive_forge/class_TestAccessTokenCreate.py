import copy
from openstackclient.identity.v3 import token
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestAccessTokenCreate(TestOAuth1):

    def setUp(self):
        super(TestAccessTokenCreate, self).setUp()
        self.access_tokens_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.OAUTH_ACCESS_TOKEN), loaded=True)
        self.cmd = token.CreateAccessToken(self.app, None)

    def test_create_access_tokens(self):
        arglist = ['--consumer-key', identity_fakes.consumer_id, '--consumer-secret', identity_fakes.consumer_secret, '--request-key', identity_fakes.request_token_id, '--request-secret', identity_fakes.request_token_secret, '--verifier', identity_fakes.oauth_verifier_pin]
        verifylist = [('consumer_key', identity_fakes.consumer_id), ('consumer_secret', identity_fakes.consumer_secret), ('request_key', identity_fakes.request_token_id), ('request_secret', identity_fakes.request_token_secret), ('verifier', identity_fakes.oauth_verifier_pin)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.access_tokens_mock.create.assert_called_with(identity_fakes.consumer_id, identity_fakes.consumer_secret, identity_fakes.request_token_id, identity_fakes.request_token_secret, identity_fakes.oauth_verifier_pin)
        collist = ('expires', 'id', 'key', 'secret')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.access_token_expires, identity_fakes.access_token_id, identity_fakes.access_token_id, identity_fakes.access_token_secret)
        self.assertEqual(datalist, data)