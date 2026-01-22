import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
class TestBarbicanWithKeystoneTokenAuth(keystone_client_fixtures.KeystoneClientFixture):

    def setUp(self):
        super(TestBarbicanWithKeystoneTokenAuth, self).setUp()
        self.test_arguments = {'--os-auth-token': 'some_token', '--os-project-id': 'some_project_id'}