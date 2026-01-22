import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
class TestBarbicanWithKeystonePasswordAuth(keystone_client_fixtures.KeystoneClientFixture):

    def setUp(self):
        super(TestBarbicanWithKeystonePasswordAuth, self).setUp()
        self.test_arguments = {'--os-username': 'some_user', '--os-password': 'some_pass'}