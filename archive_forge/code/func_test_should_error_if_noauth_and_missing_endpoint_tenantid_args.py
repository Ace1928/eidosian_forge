import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def test_should_error_if_noauth_and_missing_endpoint_tenantid_args(self):
    self._expect_error_with_invalid_noauth_args('--no-auth secret list')
    self._expect_error_with_invalid_noauth_args('--no-auth --endpoint http://xyz secret list')
    self._expect_error_with_invalid_noauth_args('--no-auth --os-tenant-id 123 secret list')
    self._expect_error_with_invalid_noauth_args('--no-auth --os-project-id 123 secret list')