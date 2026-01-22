import copy
from openstackclient.identity.v3 import token
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestOAuth1(identity_fakes.TestOAuth1):

    def setUp(self):
        super(TestOAuth1, self).setUp()
        identity_client = self.app.client_manager.identity
        self.access_tokens_mock = identity_client.oauth1.access_tokens
        self.access_tokens_mock.reset_mock()
        self.request_tokens_mock = identity_client.oauth1.request_tokens
        self.request_tokens_mock.reset_mock()
        self.projects_mock = identity_client.projects
        self.projects_mock.reset_mock()
        self.roles_mock = identity_client.roles
        self.roles_mock.reset_mock()