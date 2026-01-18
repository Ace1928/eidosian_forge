import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def test_should_fail_create_client_with_no_auth_url(self):
    args = '--os-auth-token 1234567890 --os-tenant-id 123'
    message = 'ERROR: please specify --os-auth-url'
    argv, remainder = self.parser.parse_known_args(args.split())
    e = self.assertRaises(Exception, self.barbican.create_client, argv)
    self.assertIn(message, str(e))