from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_target_user_domain_id(self, client_mock):
    self.shell('--os-target-user-domain-id=default workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('default', params[1]['target_user_domain_id'])