import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def test_should_succeed_if_noauth_with_valid_args_specified(self):
    args = '--no-auth --endpoint {0} --os-tenant-id {1}secret list'.format(self.endpoint, self.project_id)
    list_secrets_url = '{0}/v1/secrets'.format(self.endpoint.rstrip('/'))
    self.responses.get(list_secrets_url, json={'secrets': [], 'total': 0})
    client = self.create_and_assert_client(args)
    secret_list = client.secrets.list()
    self.assertTrue(self.responses._adapter.called)
    self.assertEqual(2, self.responses._adapter.call_count)
    self.assertEqual([], secret_list)