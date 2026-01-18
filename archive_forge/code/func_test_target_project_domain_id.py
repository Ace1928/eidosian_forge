from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_target_project_domain_id(self, client_mock):
    self.shell('--os-target-project-domain-id=default workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('default', params[1]['target_project_domain_id'])