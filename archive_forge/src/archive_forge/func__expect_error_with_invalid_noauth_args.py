import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def _expect_error_with_invalid_noauth_args(self, args):
    expected_err_msg = 'ERROR: please specify --endpoint and --os-project-id (or --os-tenant-id)'
    self.assert_client_raises(args, expected_err_msg)