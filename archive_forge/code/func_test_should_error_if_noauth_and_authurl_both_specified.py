import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def test_should_error_if_noauth_and_authurl_both_specified(self):
    args = '--no-auth --os-auth-url http://localhost:5000/v3'
    message = 'ERROR: argument --os-auth-url/-A: not allowed with argument --no-auth/-N'
    self.assert_client_raises(args, message)